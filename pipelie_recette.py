import asyncio
import os
import re
import json
import time
import argparse
import unicodedata
from collections import Counter
from datetime import datetime
from json import JSONDecodeError, JSONDecoder

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai import APIError, APIStatusError, RateLimitError
from tqdm import tqdm

# Ordre par defaut: modele le plus performant d'abord, puis les fallback si il echoue.
DEFAULT_MODELS = [
    "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-4o-mini",
    "mistralai/mistral-small-3.1-24b-instruct",
]
# les colonnes attendues entree et les colonnes + les sorties
REQUIRED_COLUMNS = {"id", "title", "instruction", "ingredient"}
REQUIRED_OUTPUT_KEYS = [
    "id",
    "variante_principale",
    "variante_ingredients",
    "variante_permutation1",
]
MODEL_DISABLE_400_THRESHOLD = 3
SYSTEM_JSON_INSTRUCTION = (
    "You generate culinary action variants. "
    "Return exactly one valid JSON object. No markdown. No extra text."
)

KEY_ALIASES = {
    "variante_principale": [
        "variante_principale",
        "variante_principales",
        "main_variant",
        "main_sequence",
        "primary_variant",
    ],
    "variante_ingredients": [
        "variante_ingredients",
        "ingredient_variant",
        "ingredients_variant",
        "ingredient_sequence",
    ],
    "variante_permutation1": [
        "variante_permutation1",
        "variante_permutation",
        "permutation1",
        "permutation_1",
        "variant_permutation1",
    ],
}

VERB_PATTERN = re.compile(r"^[a-z][a-z-]*$")

TERMINAL_ACTIONS = {"serve", "enjoy"}
LATE_FINISH_ACTIONS = {"garnish", "frost", "drizzle"}
COOK_ACTIONS = {
    "bake",
    "boil",
    "cook",
    "fry",
    "grill",
    "heat",
    "microwave",
    "poach",
    "roast",
    "saute",
    "sear",
    "simmer",
    "steam",
    "toast",
}
OVEN_STYLE_ACTIONS = {"bake", "grill", "roast", "toast"}
STRUCTURE_BEFORE_COOK_ACTIONS = {"fill", "stuff", "shape", "drop"}
COOLING_AFTER_COOK_ACTIONS = {"cool"}
CRITICAL_ORDER_ACTIONS = (
    TERMINAL_ACTIONS
    | LATE_FINISH_ACTIONS
    | COOLING_AFTER_COOK_ACTIONS
    | {"preheat"}
    | COOK_ACTIONS
)


class OutputValidationError(Exception):
    pass


def normalize_recipe_id(value):
    return str(value).strip()


def canonicalize_verb(action):
    text = str(action).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _coerce_to_action_list(value):
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        raw = re.split(r"[\n;,]+", value)
    else:
        raw = []
    return [canonicalize_verb(v) for v in raw if canonicalize_verb(v)]


def _pick_value_with_aliases(source, target_key):
    for alias in KEY_ALIASES[target_key]:
        if alias in source:
            return source[alias]
    return None


def build_prompt(row, reject_reason=None):
    rid = normalize_recipe_id(row["id"])
    title = str(row["title"])
    instruction = str(row["instruction"])
    ingredient = str(row["ingredient"])

    reject_block = ""
    if reject_reason:
        reject_block = (
            "Previous output was rejected: "
            f"{reject_reason}. Regenerate a valid output.\n\n"
        )

    return (
        f"{reject_block}"
        "Task: generate 3 action variants for this recipe.\n\n"
        f"ID: {rid}\n"
        f"TITLE: {title}\n"
        f"INSTRUCTIONS: {instruction}\n"
        f"INGREDIENTS: {ingredient}\n\n"
        "STRICT RULES:\n"
        "1) Return EXACTLY one JSON object.\n"
        "2) Use only these keys: id, variante_principale, variante_ingredients, variante_permutation1.\n"
        "3) Each action must be ONE short lowercase verb (e.g., mix, chop, pour, heat).\n"
        "4) Forbidden: long phrases, punctuation, quantities, temperatures.\n"
        "5) Minimum 3 actions per list, no maximum limit.\n"
        "5b) Each action verb must appear only once per list (no duplicates).\n"
        "6) variante_principale = canonical workflow.\n"
        "7) variante_ingredients = ingredient-focused workflow; must have AT LEAST as many actions as variante_principale.\n"
        "8) variante_permutation1 = same verbs as variante_principale, different order.\n\n"
        "9) Keep chronological cooking logic: prep -> mix -> cook -> finish -> serve.\n"
        "10) Never place serve/enjoy before cooking actions.\n\n"
        "Schema:\n"
        "{\n"
        f"  \"id\": \"{rid}\",\n"
        "  \"variante_principale\": [\"mix\", \"heat\", \"serve\"],\n"
        "  \"variante_ingredients\": [\"chop\", \"measure\", \"pour\"],\n"
        "  \"variante_permutation1\": [\"heat\", \"mix\", \"serve\"]\n"
        "}\n\n"
        "Return ONLY this JSON."
    )


def build_messages(row, reject_reason=None):
    return [
        {"role": "system", "content": SYSTEM_JSON_INSTRUCTION},
        {"role": "user", "content": build_prompt(row, reject_reason=reject_reason)},
    ]


def build_repair_messages(row, raw_text, reason):
    rid = normalize_recipe_id(row["id"])
    schema = (
        "{\n"
        f"  \"id\": \"{rid}\",\n"
        "  \"variante_principale\": [\"mix\", \"heat\", \"serve\"],\n"
        "  \"variante_ingredients\": [\"chop\", \"measure\", \"pour\"],\n"
        "  \"variante_permutation1\": [\"heat\", \"mix\", \"serve\"]\n"
        "}"
    )
    return [
        {"role": "system", "content": SYSTEM_JSON_INSTRUCTION},
        {
            "role": "user",
            "content": (
                "Fix the following output so it matches the exact schema.\n"
                f"Rejection reason: {reason}\n\n"
                "Rules reminder:\n"
                "- 1 action = 1 lowercase verb\n"
                "- no extra keys\n"
                "- no duplicate verbs in the same list\n"
                "- variante_permutation1 must be a permutation of variante_principale\n"
                "- variante_ingredients must have at least as many actions as variante_principale\n"
                "- keep chronological cooking logic and keep serve/enjoy at the end\n\n"
                f"Schema:\n{schema}\n\n"
                f"Output to fix:\n{raw_text}"
            ),
        },
    ]


def extract_text_content(response):
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content or "")


def extract_json_object(text):
    if text is None:
        raise JSONDecodeError("Empty response", "", 0)

    raw_text = str(text).strip()
    if not raw_text:
        raise JSONDecodeError("Empty response", raw_text, 0)

    fenced = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", raw_text, flags=re.IGNORECASE)
    if fenced:
        raw_text = fenced.group(1).strip()

    try:
        obj = json.loads(raw_text)
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            return obj[0]
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    decoder = JSONDecoder()
    for i, ch in enumerate(raw_text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw_text[i:])
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
                return obj[0]
        except JSONDecodeError:
            continue

    raise JSONDecodeError("No JSON object found", raw_text, 0)


def _validate_action_list(actions, key):
    if not actions:
        raise OutputValidationError(f"Missing/invalid key: {key}")

    if len(actions) < 3:
        raise OutputValidationError(f"Too few actions for {key}: {len(actions)}")

    duplicates = [v for v, c in Counter(actions).items() if c > 1]
    if duplicates:
        raise OutputValidationError(f"Duplicate actions in {key}: {', '.join(duplicates)}")

    if len(set(actions)) < 2:
        raise OutputValidationError(f"Low-information actions for {key}")

    for action in actions:
        if not VERB_PATTERN.fullmatch(action):
            raise OutputValidationError(f"Invalid verb format in {key}: {action}")


def _first_index(actions, candidates):
    indices = [i for i, action in enumerate(actions) if action in candidates]
    if not indices:
        return None
    return min(indices)


def _last_index(actions, candidates):
    indices = [i for i, action in enumerate(actions) if action in candidates]
    if not indices:
        return None
    return max(indices)


def _assert_before(actions, left_set, right_set, key, message):
    left_last = _last_index(actions, left_set)
    right_first = _first_index(actions, right_set)
    if left_last is not None and right_first is not None and left_last > right_first:
        raise OutputValidationError(f"Logical order error in {key}: {message}")


def _validate_logical_order(actions, key):
    n = len(actions)

    for terminal in TERMINAL_ACTIONS:
        terminal_positions = [i for i, a in enumerate(actions) if a == terminal]
        if terminal_positions and any(pos != n - 1 for pos in terminal_positions):
            raise OutputValidationError(
                f"Logical order error in {key}: '{terminal}' must be the final action"
            )

    _assert_before(
        actions,
        {"preheat"},
        OVEN_STYLE_ACTIONS,
        key,
        "preheat must happen before bake/grill/roast/toast",
    )
    _assert_before(
        actions,
        COOK_ACTIONS,
        TERMINAL_ACTIONS,
        key,
        "serve/enjoy must happen after cooking actions",
    )


def _validate_permutation_critical_order(principale, permutation):
    principale_pos = {}
    permutation_pos = {}
    for i, action in enumerate(principale):
        if action in CRITICAL_ORDER_ACTIONS and action not in principale_pos:
            principale_pos[action] = i
    for i, action in enumerate(permutation):
        if action in CRITICAL_ORDER_ACTIONS and action not in permutation_pos:
            permutation_pos[action] = i

    shared = [a for a in principale_pos if a in permutation_pos]
    shared = sorted(shared, key=lambda a: principale_pos[a])

    for i, action_a in enumerate(shared):
        for action_b in shared[i + 1 :]:
            if permutation_pos[action_a] > permutation_pos[action_b]:
                raise OutputValidationError(
                    "Logical order error in variante_permutation1: "
                    f"critical order reversed ({action_a} before {action_b})"
                )


def _to_single_verb_list(value):
    actions = _coerce_to_action_list(value)
    verbs = []
    for action in actions:
        first = action.split(" ")[0].strip() if action else ""
        if first:
            verbs.append(first)
    return verbs


def _build_safe_permutation(principale):
    n = len(principale)
    if n < 2:
        return None

    for i in range(n):
        for j in range(i + 1, n):
            if principale[i] == principale[j]:
                continue
            candidate = principale[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if candidate == principale:
                continue
            try:
                _validate_logical_order(candidate, "variante_permutation1")
                _validate_permutation_critical_order(principale, candidate)
                return candidate
            except OutputValidationError:
                continue
    return None


def try_local_autofix(raw_obj, row):
    if not isinstance(raw_obj, dict):
        return None

    for wrapper_key in ("data", "result", "output"):
        inner = raw_obj.get(wrapper_key)
        if isinstance(inner, dict):
            raw_obj = inner
            break

    rid = normalize_recipe_id(row["id"])
    fixed = {"id": rid}

    for key in REQUIRED_OUTPUT_KEYS[1:]:
        value = _pick_value_with_aliases(raw_obj, key)
        verbs = _to_single_verb_list(value)
        # Retirer les doublons en gardant l'ordre
        seen = set()
        unique_verbs = []
        for v in verbs:
            if v not in seen:
                seen.add(v)
                unique_verbs.append(v)
        fixed[key] = unique_verbs

    for key in REQUIRED_OUTPUT_KEYS[1:]:
        if len(fixed[key]) < 3:
            return None

    principale = fixed["variante_principale"]
    permutation = fixed["variante_permutation1"]
    need_rebuild = False
    if len(permutation) != len(principale):
        need_rebuild = True
    elif Counter(permutation) != Counter(principale):
        need_rebuild = True
    elif permutation == principale:
        need_rebuild = True
    else:
        try:
            _validate_logical_order(permutation, "variante_permutation1")
            _validate_permutation_critical_order(principale, permutation)
        except OutputValidationError:
            need_rebuild = True

    if need_rebuild:
        rebuilt = _build_safe_permutation(principale)
        if rebuilt is None:
            return None
        fixed["variante_permutation1"] = rebuilt

    try:
        return validate_and_normalize_output(fixed, row)
    except OutputValidationError:
        return None


def validate_and_normalize_output(raw_obj, row):
    if not isinstance(raw_obj, dict):
        raise OutputValidationError("Model output is not a JSON object")

    for wrapper_key in ("data", "result", "output"):
        inner = raw_obj.get(wrapper_key)
        if isinstance(inner, dict):
            raw_obj = inner
            break

    required_keys = set(REQUIRED_OUTPUT_KEYS)
    if set(raw_obj.keys()) != required_keys:
        raise OutputValidationError("Invalid keys: output must contain exactly required keys")

    rid = normalize_recipe_id(row["id"])
    output_id = normalize_recipe_id(raw_obj.get("id", ""))
    if output_id != rid:
        raise OutputValidationError(f"ID mismatch: expected {rid}, got {output_id}")

    normalized = {"id": rid}
    for key in REQUIRED_OUTPUT_KEYS[1:]:
        value = _pick_value_with_aliases(raw_obj, key)
        actions = _coerce_to_action_list(value)
        _validate_action_list(actions, key)
        _validate_logical_order(actions, key)
        normalized[key] = actions

    if normalized["variante_principale"] == normalized["variante_ingredients"]:
        raise OutputValidationError("variante_ingredients must differ from variante_principale")

    if len(normalized["variante_ingredients"]) < len(normalized["variante_principale"]):
        raise OutputValidationError(
            f"variante_ingredients must have at least as many actions as variante_principale "
            f"({len(normalized['variante_ingredients'])} < {len(normalized['variante_principale'])})"
        )

    principale = normalized["variante_principale"]
    permutation = normalized["variante_permutation1"]
    if sorted(permutation) != sorted(principale):
        raise OutputValidationError(
            "variante_permutation1 must be a permutation of variante_principale"
        )
    if permutation == principale:
        raise OutputValidationError("variante_permutation1 must have a different order")
    _validate_permutation_critical_order(principale, permutation)

    return normalized


def load_input(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".json"):
        df = pd.read_json(path)
    else:
        raise ValueError("Input must be .csv or .json")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_str}")

    df["id"] = df["id"].map(normalize_recipe_id)
    return df


def load_checkpoint(path):
    if not path or not os.path.exists(path):
        return {}, set()

    data = {}
    malformed_lines = 0
    invalid_quality_lines = 0
    missing_required_lines = 0
    extra_key_lines = 0
    required_keys = set(REQUIRED_OUTPUT_KEYS)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                keys = set(obj.keys())
                if not required_keys.issubset(keys):
                    missing_required_lines += 1
                    continue
                if keys != required_keys:
                    extra_key_lines += 1
                    continue

                rid = normalize_recipe_id(obj.get("id", ""))
                if not rid:
                    malformed_lines += 1
                    continue

                validation_row = {"id": rid, "title": "", "instruction": "", "ingredient": ""}
                validated = validate_and_normalize_output(obj, validation_row)
                data[rid] = {
                    "id": rid,
                    "variante_principale": validated["variante_principale"],
                    "variante_ingredients": validated["variante_ingredients"],
                    "variante_permutation1": validated["variante_permutation1"],
                }
            except OutputValidationError:
                invalid_quality_lines += 1
            except Exception:
                malformed_lines += 1

    if malformed_lines:
        print(f"Checkpoint warning: {malformed_lines} malformed lines ignored")
    if invalid_quality_lines:
        print(f"Checkpoint info: {invalid_quality_lines} invalid rows ignored and will be reprocessed")
    if missing_required_lines:
        print(f"Checkpoint info: {missing_required_lines} rows missing required keys ignored and will be reprocessed")
    if extra_key_lines:
        print(f"Checkpoint info: {extra_key_lines} rows with extra keys ignored and will be reprocessed")

    return data, set(data.keys())


def append_checkpoint(path, obj):
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_failed_recipe(path, source_row, reason):
    if not path:
        return
    payload = {
        "id": normalize_recipe_id(source_row["id"]),
        "title": str(source_row.get("title", "")),
        "instruction": str(source_row.get("instruction", "")),
        "ingredient": str(source_row.get("ingredient", "")),
        "error": str(reason),
        "failed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _build_result_row(source_row, output_obj):
    return {
        "id": normalize_recipe_id(source_row["id"]),
        "variante_principale": output_obj["variante_principale"],
        "variante_ingredients": output_obj["variante_ingredients"],
        "variante_permutation1": output_obj["variante_permutation1"],
    }


def resolve_models(models=None):
    if models:
        return models

    env_models = os.getenv("PIPELINE_MODELS", "").strip()
    if env_models:
        parsed = [m.strip() for m in env_models.split(",") if m.strip()]
        if parsed:
            return parsed

    return DEFAULT_MODELS


# ──────────────────────────────────────────────
# ASYNC PIPELINE
# ──────────────────────────────────────────────

async def _call_llm_async(client, model, messages, max_tokens=800, temperature=0.0, timeout_seconds=60):
    return await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout_seconds,
    )


async def _parse_or_repair_async(client, model, row, raw_content, timeout_seconds, max_tokens):
    parsed = None
    try:
        parsed = extract_json_object(raw_content)
        return validate_and_normalize_output(parsed, row)
    except OutputValidationError as err:
        local_fixed = try_local_autofix(parsed, row)
        if local_fixed is not None:
            return local_fixed
        repair_messages = build_repair_messages(row, raw_content, str(err))
        repaired = await _call_llm_async(
            client, model, repair_messages,
            max_tokens=max_tokens, timeout_seconds=timeout_seconds,
        )
        repaired_content = extract_text_content(repaired)
        parsed_repaired = extract_json_object(repaired_content)
        try:
            return validate_and_normalize_output(parsed_repaired, row)
        except OutputValidationError:
            local_fixed_repaired = try_local_autofix(parsed_repaired, row)
            if local_fixed_repaired is not None:
                return local_fixed_repaired
            raise
    except JSONDecodeError as err:
        repair_messages = build_repair_messages(row, raw_content, str(err))
        repaired = await _call_llm_async(
            client, model, repair_messages,
            max_tokens=max_tokens, timeout_seconds=timeout_seconds,
        )
        repaired_content = extract_text_content(repaired)
        parsed_repaired = extract_json_object(repaired_content)
        try:
            return validate_and_normalize_output(parsed_repaired, row)
        except OutputValidationError:
            local_fixed_repaired = try_local_autofix(parsed_repaired, row)
            if local_fixed_repaired is not None:
                return local_fixed_repaired
            raise


async def _process_recipe_async(
    row, client, models, semaphore, lock,
    done_ids, results, failed_ids,
    checkpoint_path, failed_path,
    max_retries, max_tokens, request_timeout_seconds,
    disabled_models, model_400_counts,
    pbar,
):
    rid = normalize_recipe_id(row["id"])

    async with lock:
        if rid in done_ids:
            pbar.update(1)
            return

    async with semaphore:
        success = False
        last_error = "unknown_error"

        for model in models:
            if model in disabled_models:
                continue

            reject_reason = None

            for retry in range(max_retries):
                try:
                    messages = build_messages(row, reject_reason=reject_reason)
                    resp = await _call_llm_async(
                        client, model, messages,
                        max_tokens=max_tokens,
                        temperature=0.0,
                        timeout_seconds=request_timeout_seconds,
                    )
                    content = extract_text_content(resp)
                    output_obj = await _parse_or_repair_async(
                        client, model, row, content,
                        request_timeout_seconds, max_tokens,
                    )

                    result_row = _build_result_row(row, output_obj)

                    async with lock:
                        results.append(result_row)
                        done_ids.add(rid)
                        append_checkpoint(checkpoint_path, result_row)
                        model_400_counts[model] = 0

                    success = True
                    break

                except OutputValidationError as e:
                    reject_reason = str(e)
                    last_error = f"{model}: {e}"
                    print(f"ID {rid} invalid output [{model}] retry {retry + 1}/{max_retries}: {e}")
                    await asyncio.sleep(2 ** retry)

                except JSONDecodeError:
                    reject_reason = "json_parse_error"
                    last_error = f"{model}: json_parse_error"
                    await asyncio.sleep(2 ** retry)

                except RateLimitError:
                    last_error = f"{model}: rate_limit"
                    await asyncio.sleep(2 ** retry)

                except APIStatusError as e:
                    code = getattr(e, "status_code", None)
                    last_error = f"{model}: api_status_{code}"

                    if code == 402:
                        break
                    if code == 404:
                        disabled_models.add(model)
                        break
                    if code == 429:
                        await asyncio.sleep(2 ** retry)
                        continue
                    if code == 400:
                        model_400_counts[model] += 1
                        if model_400_counts[model] >= MODEL_DISABLE_400_THRESHOLD:
                            disabled_models.add(model)
                            print(f"Model disabled after repeated 400: {model}")
                        break
                    break

                except APIError as e:
                    error_str = str(e).lower()
                    if "connection" in error_str or "timeout" in error_str:
                        print(f"ID {rid} connection error [{model}] retry {retry + 1}/{max_retries}")
                        await asyncio.sleep(max(5, 2 ** retry))
                        continue
                    last_error = f"{model}: api_error {e}"
                    break

                except Exception as e:
                    last_error = f"{model}: unexpected {e}"
                    await asyncio.sleep(2 ** retry)

            if success:
                break

        if not success:
            async with lock:
                failed_ids.append(rid)
                append_failed_recipe(failed_path, row, last_error)
            print(f"ID {rid} echec des modeles")

    pbar.update(1)


async def _pipeline_async(
    df_final_clean,
    client,
    models,
    max_concurrent=10,
    max_retries=3,
    checkpoint_path="data_checkpoint.jsonl",
    failed_path="data_echouee.jsonl",
    request_timeout_seconds=60,
    log_every=100,
    max_tokens=800,
):
    missing = REQUIRED_COLUMNS - set(df_final_clean.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df = df_final_clean.copy()
    df["id"] = df["id"].map(normalize_recipe_id)

    print(f"PIPELINE VARIANTS - {len(df)} flags")
    print(f"Modeles: {', '.join(models)}")
    print(f"Concurrence: {max_concurrent} requetes simultanees\n")

    results = []
    failed_ids = []
    disabled_models = set()
    model_400_counts = {model: 0 for model in models}
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_concurrent)

    checkpoint_data, done_ids = load_checkpoint(checkpoint_path)
    results.extend(checkpoint_data.values())
    if done_ids:
        print(f"Reprise: {len(done_ids)} deja traites")

    rows_to_process = [
        row for _, row in df.iterrows()
        if normalize_recipe_id(row["id"]) not in done_ids
    ]

    with tqdm(total=len(df), initial=len(done_ids), desc="Recettes") as pbar:
        tasks = [
            _process_recipe_async(
                row, client, models, semaphore, lock,
                done_ids, results, failed_ids,
                checkpoint_path, failed_path,
                max_retries, max_tokens, request_timeout_seconds,
                disabled_models, model_400_counts,
                pbar,
            )
            for row in rows_to_process
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    print(f"FAILED: {len(failed_ids)}")
    return pd.DataFrame(results)


def run_from_notebook(
    df_final_clean,
    models=None,
    max_concurrent=10,
    max_retries=3,
    checkpoint_path="data_checkpoint.jsonl",
    failed_path="data_echouee.jsonl",
    output_prefix="data_variantes",
    api_key=None,
    base_url="https://openrouter.ai/api/v1",
    save_outputs=True,
    request_timeout_seconds=60,
    log_every=100,
    max_tokens=800,
    # parametres legacy ignores (compatibilite notebooks precedents)
    batch_size=None,
    sleep_per_batch=None,
    sleep_per_request=None,
):
    load_dotenv()

    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing (.env or environment variable required).")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    models = resolve_models(models)

    coro = _pipeline_async(
        df_final_clean,
        client=client,
        models=models,
        max_concurrent=max_concurrent,
        max_retries=max_retries,
        checkpoint_path=checkpoint_path,
        failed_path=failed_path,
        request_timeout_seconds=request_timeout_seconds,
        log_every=log_every,
        max_tokens=max_tokens,
    )

    try:
        loop = asyncio.get_running_loop()
        # Deja dans un event loop (Jupyter) — on applique nest_asyncio
        import nest_asyncio
        nest_asyncio.apply()
        df_results = loop.run_until_complete(coro)
    except RuntimeError:
        df_results = asyncio.run(coro)

    if not save_outputs:
        return df_results

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = f"{output_prefix}_{timestamp}.csv"
    json_path = f"{output_prefix}_{timestamp}.json"

    df_results.to_csv(csv_path, index=False)
    df_results.to_json(json_path, orient="records", indent=2)

    return df_results, csv_path, json_path, checkpoint_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="df_final_clean.csv")
    parser.add_argument("--models", default="")
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--checkpoint", default="data_checkpoint.jsonl")
    parser.add_argument("--failed-path", default="data_echouee.jsonl")
    parser.add_argument("--output-prefix", default="data_variantes")
    parser.add_argument("--request-timeout-seconds", type=float, default=60)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=800)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing (.env or environment variable required).")

    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    df_input = load_input(args.input)

    cli_models = [m.strip() for m in args.models.split(",") if m.strip()]
    models = resolve_models(cli_models if cli_models else None)

    df_results = asyncio.run(
        _pipeline_async(
            df_input,
            client=client,
            models=models,
            max_concurrent=args.max_concurrent,
            max_retries=args.max_retries,
            checkpoint_path=args.checkpoint,
            failed_path=args.failed_path,
            request_timeout_seconds=args.request_timeout_seconds,
            log_every=args.log_every,
            max_tokens=args.max_tokens,
        )
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = f"{args.output_prefix}_{timestamp}.csv"
    json_path = f"{args.output_prefix}_{timestamp}.json"
    df_results.to_csv(csv_path, index=False)
    df_results.to_json(json_path, orient="records", indent=2)

    print(f"TERMINE: {len(df_results)}/{len(df_input)} reussis")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Echecs: {args.failed_path}")


if __name__ == "__main__":
    main()
