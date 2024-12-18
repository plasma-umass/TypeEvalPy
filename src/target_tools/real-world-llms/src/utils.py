import json
import os
import re
import shutil
import sys
import yaml

import requests
import logging
import prompts
import copy
import tiktoken
import csv

logger = logging.getLogger("runner")
logger.setLevel(logging.DEBUG)


class JsonException(Exception):
    pass


class TimeoutException(Exception):
    pass


def is_ollama_online(server_url):
    try:
        res = requests.get(server_url)
        # Check if the request was successful
        if res.status_code == 200:
            # Check the content of the response
            if res.text == "Ollama is running":
                return True
        return False
    except requests.exceptions.RequestException as e:
        # Handle any exceptions that occur during the request
        print(f"An error occurred: {e}")
        return False


def copy_folder(src, dst):
    """
    Copies a folder from the source (src) to the destination (dst).

    :param src: Source folder path
    :param dst: Destination folder path
    """
    # Check if the source directory exists
    if not os.path.exists(src):
        print(f"Source folder {src} does not exist.")
        return

    # Check if the destination directory exists, if so, remove it
    if os.path.exists(dst):
        shutil.rmtree(dst)
        print(f"Existing folder at {dst} has been removed.")

    # Copy the folder
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"Folder copied from {src} to {dst}")


def is_running_in_docker():
    """Check if Python is running inside a Docker container."""
    return (
        os.path.exists("/.dockerenv")
        or os.environ.get(  # Check if the /.dockerenv file exists
            "DOCKER_CONTAINER", False
        )
        or os.environ.get(  # Check if DOCKER_CONTAINER environment variable is set
            "DOCKER_IMAGE_NAME", False
        )  # Check if DOCKER_IMAGE_NAME environment variable is set
    )


def generate_json_file(filename, type_info):
    # Generate JSON file with type information
    try:
        if isinstance(type_info, list):
            pass
        else:
            type_info = json.loads(type_info)
        is_valid_json = True
    except Exception as e:
        is_valid_json = False
        print(f"Not a valid JSON: {e}")

    json_data = json.dumps(type_info, indent=4)
    with open(filename, "w") as file:
        file.write(json_data)

    return is_valid_json


def generate_json_from_answers(repo, gt_json_file, answers):
    try:
        with open(gt_json_file, "r") as file:
            gt_data = json.load(file)

        pattern = re.compile(r"^\s*(\d+)\.\s+(.+)\s*$", re.MULTILINE)
        parsed_answers = pattern.findall(answers)

        parsed_answers = {int(x) - 1: y for x, y in parsed_answers}
        # if len(gt_data) != len(parsed_answers):
        #     return []

        # Filter gt_data to only include instances where the file name matches the repo
        repo_gt_data = [entry for entry in gt_data if entry.get("file") == repo]

        answers_json_data = []
        for fact in range(len(repo_gt_data)):
            _result = repo_gt_data[fact]
            _result.pop("type")
            if fact in parsed_answers:
                _result["type"] = [x.strip() for x in parsed_answers[fact].split(",")]
                answers_json_data.append(_result)

        return answers_json_data
    except Exception as e:
        print("Error generating json from questions")
        print(e)
        return []


def generate_answers_for_fine_tuning(json_file):
    # Read and parse the JSON file
    with open(json_file, "r") as file:
        data = json.load(file)

    counter = 1
    answers = []
    for fact in data:
        answers.append(f"{counter}. {', '.join(fact['type'])}")
        counter += 1

    return "\n".join(answers)


def generate_questions_from_json(json_file):
    # Read and parse the JSON file
    with open(json_file, "r") as file:
        data = json.load(file)

    questions = []

    for entry in data:
        file = entry["file"]
        line_number = entry["line_number"]
        col_offset = entry["col_offset"]

        # Generate different questions based on the content of each entry
        # Function Return type
        if "function" in entry and "parameter" not in entry and "variable" not in entry:
            question = (
                "What is the return type of the function"
                f" '{entry['function']}' at line {line_number}, column"
                f" {col_offset}?"
            )
        # Function Parameter type
        elif "parameter" in entry:
            question = (
                f"What is the type of the parameter '{entry['parameter']}' at line"
                f" {line_number}, column {col_offset}, within the function"
                f" '{entry['function']}'?"
            )
        # Variable in a function type
        elif "variable" in entry and "function" not in entry:
            question = (
                f"What is the type of the variable '{entry['variable']}' at line"
                f" {line_number}, column {col_offset}?"
            )
        elif "variable" in entry and "function" in entry:
            question = (
                f"What is the type of the variable '{entry['variable']}' at line"
                f" {line_number}, column {col_offset}, within the function"
                f" '{entry['function']}'?"
            )
        else:
            print("ERROR! Type could not be converted to types")
        questions.append(question)

    if len(data) != len(questions):
        print("ERROR! Type questions length does not match json length")
        sys.exit(-1)

    questions = [f"{x}. {y}" for x, y in zip(range(1, len(questions) + 1), questions)]
    return questions


def load_models_config(config_path):
    models_config = {"models": {}, "custom_models": {}, "openai_models": {}}
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)
        for model_data in config_data["models"]:
            models_config["models"][model_data["name"]] = model_data
        for model_data in config_data["custom_models"]:
            models_config["custom_models"][model_data["name"]] = model_data
        for model_data in config_data["openai_models"]:
            models_config["openai_models"][model_data["name"]] = model_data

    return models_config


def load_runner_config(config_path):
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    return config_data["runner_config"]


def gather_code_files_from_test_folder(test_folder, language_extension="py"):
    """Recursively gathers all code files with the specified language extension."""
    code_files = []
    for root, _, files in os.walk(test_folder):
        for file in files:
            if file.endswith(f".{language_extension}"):
                code_files.append(os.path.join(root, file))
    return code_files


def get_token_count(text, prompt_id):
    """
    Retrieves the token count of the given text.

    Args:
        text (str): The text to be tokenized.

    Returns:
        int: The token count.
    """
    prices_per_token = {
        "gpt-3.5-turbo-0125": 0.0000005,
        "gpt-4-turbo": 0.00001,
        "gpt-4": 0.00001,
        "gpt-4o": 0.000005,
    }

    # Ensure text is a string
    if isinstance(text, list):
        # Convert list of dictionaries to list of strings
        text = " ".join(
            [str(item) if isinstance(item, dict) else item for item in text]
        )

    encoding = tiktoken.encoding_for_model("gpt-4o")
    number_of_tokens_4o = len(encoding.encode(text))
    logger.debug(
        f"Number of tokens for model `gpt-4o`: {number_of_tokens_4o}"
        + f" Cost: {number_of_tokens_4o * prices_per_token['gpt-4o']:.5f}"
        + f" Prompt: {prompt_id}"
    )

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    number_of_tokens_3_5 = len(encoding.encode(text))
    logger.debug(
        f"Number of tokens for model `gpt-3.5-turbo`: {number_of_tokens_3_5}"
        + f" Cost: {number_of_tokens_3_5 * prices_per_token['gpt-3.5-turbo-0125']:.5f}"
        + f" Prompt: {prompt_id}"
    )

    encoding = tiktoken.encoding_for_model("gpt-4")
    number_of_tokens_4 = len(encoding.encode(text))
    logger.debug(
        f"Number of tokens for model `gpt-4-turbo`: {number_of_tokens_4}"
        + f" Cost: {number_of_tokens_4 * prices_per_token['gpt-4']:.5f}"
        + f" Prompt: {prompt_id}"
    )

    return {
        "gpt-3.5-turbo": number_of_tokens_3_5,
        "gpt-4-turbo": number_of_tokens_4,
    }


def generate_questions_from_metadata(metadata):
    """
    Generates questions based on the metadata passed (previously read from JSON).
    """
    questions = []

    for entry in metadata:
        file = entry.get("file")
        line_number = entry.get("line_number", "unknown")
        col_offset = entry.get("col_offset", "unknown")

        # Ensure we have either 'function', 'parameter', or 'variable'
        if "function" in entry and "parameter" not in entry and "variable" not in entry:
            question = f"What is the return type of the function '{entry['function']}' at line {line_number}, column {col_offset}?"
        elif "parameter" in entry:
            question = f"What is the type of the parameter '{entry['parameter']}' at line {line_number}, column {col_offset}, within the function '{entry['function']}'?"
        elif "variable" in entry and "function" not in entry:
            question = f"What is the type of the variable '{entry['variable']}' at line {line_number}, column {col_offset}?"
        elif "variable" in entry and "function" in entry:
            question = f"What is the type of the variable '{entry['variable']}' at line {line_number}, column {col_offset}, within the function '{entry['function']}'?"
        else:
            print(f"ERROR! Type could not be converted to types for entry: {entry}")
            continue

        questions.append(question)

    # Number the questions
    questions = [f"{x}. {y}" for x, y in zip(range(1, len(questions) + 1), questions)]

    return questions


def generate_csv(token_counts, prompt_id):
    """
    Generates a CSV file with the token count information.

    Args:
        token_counts (dict): Token counts for different models.
        prompt_id (str): Identifier for the prompt template.
    """
    csv_file = "prompt_token_counts.csv"
    fieldnames = ["prompt_id", "model", "token_count"]

    # Check if the file exists
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header only if the file doesn't exist
        if not file_exists:
            writer.writeheader()

        for model, token_count in token_counts.items():
            writer.writerow(
                {"prompt_id": prompt_id, "model": model, "token_count": token_count}
            )


def get_prompt(
    prompt_id,
    source_code,
    metadata=None,
    answers_placeholders=True,
    use_system_prompt=True,
):
    """
    Generates a prompt based on the given prompt_id, metadata, and file path.

    Args:
        prompt_id (str): Identifier for the prompt template.
        file_path (str): Path to the file associated with the prompt.
        metadata (list): Metadata used to generate questions (if applicable).
        answers_placeholders (bool): Whether to include placeholder answers.
        use_system_prompt (bool): Whether to include the system prompt.

    Returns:
        dict: The generated prompt.
    """
    if prompt_id in ["prompt_template_questions_based_2"]:
        # Use metadata to generate questions
        if metadata is None:
            raise ValueError("Metadata is required for this prompt template.")

        questions_from_metadata = generate_questions_from_metadata(metadata)
        prompt_data = {
            "code": source_code,
            "questions": "\n".join(questions_from_metadata),
            "answers": (
                "\n".join([f"{x}." for x in range(1, len(questions_from_metadata) + 1)])
                if answers_placeholders
                else ""
            ),
        }

        if use_system_prompt:
            prompt = copy.deepcopy(eval(f"prompts.{prompt_id}"))
            prompt[1]["content"] = prompt[1]["content"].format(**prompt_data)
        else:
            prompt = copy.deepcopy(eval(f"prompts.{prompt_id}_no_sys"))
            prompt[0]["content"] = prompt[0]["content"].format(**prompt_data)

    elif prompt_id in ["prompt_template_masked_code_based_1"]:
        json_filepath = str(source_code).replace(".py", "_gt.json")
        test_dir = os.path.dirname(json_filepath)
        code_files = gather_code_files_from_test_folder(test_dir)

        # Concatenate code contents with masked file input
        code = ""
        for code_file in code_files:
            try:
                with open(code_file, "r") as file:
                    masked_code_content = (
                        file.read()
                    )  # Assuming files are already masked
                    relative_path = os.path.relpath(code_file, test_dir)
                    # Add filename to the code content for context
                    code += f"```{relative_path}\n{masked_code_content}```\n\n"
            except FileNotFoundError:
                logger.warning(f"Code file {code_file} not found. Skipping.")

        prompt_data = {
            "code": code,
            "instructions": (
                "You are given a Python code snippet where all type annotations are currently represented by the placeholder '[MASK]'. "
                "Your task is to replace '[MASK]' with the most appropriate Python type annotations, such as 'str', 'int', 'callable', etc., "
                "for all function return types, variable annotations, and function parameters. "
                "\n\nStrict Requirements:\n"
                "1. Maintain the exact same structure, formatting, and indentation as in the input code.\n"
                "2. Do not alter the line numbers or remove existing blank lines.\n"
                "3. Do not add any additional blank lines or comments.\n"
                "4. Do not add any explanations or extra information in the output.\n"
                "5. Only return the annotated version of the code.\n"
                "6. Ensure proper and consistent type annotations wherever applicable."
            ),
        }

        if use_system_prompt:
            prompt = copy.deepcopy(eval(f"prompts.{prompt_id}"))
            prompt[1]["content"] = "{instructions}\n\n{code}".format(**prompt_data)
        else:
            prompt = copy.deepcopy(eval(f"prompts.{prompt_id}_no_sys"))
            prompt[0]["content"] = "{instructions}\n\n{code}".format(**prompt_data)

    else:
        raise ValueError(f"Unknown prompt_id: {prompt_id}")

    # Calculate token count
    token_counts = get_token_count(prompt, prompt_id)

    # Generate CSV
    generate_csv(token_counts, prompt_id)

    return prompt

def dump_ft_jsonl(id_mapping, output_file):
    mappings = copy.deepcopy(id_mapping)
    for _m in mappings.values():
        assistant_message = {
            "role": "assistant",
            "content": generate_answers_for_fine_tuning(_m["json_filepath"]),
        }
        _m["prompt"].append(assistant_message)

    prompts = [x["prompt"] for x in mappings.values()]

    with open(output_file, "w") as output:
        for _m in prompts:
            output.write(json.dumps(_m))
            output.write("\n")


def dump_batch_prompt_jsonl(
    id_mapping, output_file, id_prefix="types", model="gpt-4o-mini"
):
    with open(output_file, "w") as output:
        for idx, _m in id_mapping.items():
            prompt_dict = {
                "custom_id": f"request-{id_prefix}-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": _m["prompt"],
                    "max_tokens": 250,
                },
            }
            output.write(json.dumps(prompt_dict))
            output.write("\n")


def get_prompt_cost(prompts):
    """
    Retrieves the token count of the given text.

    Args:
        text (str): The text to be tokenized.

    Returns:
        int: The token count.
    """

    prices_per_token = {
        "gpt-4o": 0.000005,
        "gpt-4o-mini": 0.00000015,
    }

    for model, price in prices_per_token.items():
        encoding = tiktoken.encoding_for_model(model)
        number_of_tokens = len(encoding.encode(str(prompts)))
        logger.info(
            f"Number of tokens for model `{model}`: {number_of_tokens}"
            + f" Cost: {number_of_tokens * price:.5f}"
        )


# Example usage:
# loader = ConfigLoader("models_config.yaml")
# loader.load_config()
# models = loader.get_models()
# for model in models:
#     print(model.name, model.model_path)
