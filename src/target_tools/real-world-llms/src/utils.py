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
from multiprocessing import Pool
from tqdm import tqdm

logger = logging.getLogger("runner")
logger.setLevel(logging.INFO)

# Initialize counters
exceeded_limit_count = 0
within_limit_count = 0


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
    If the destination folder exists, it retains its content and continues.
    If the destination folder does not exist, it is created.

    :param src: Source folder path
    :param dst: Destination folder path
    """
    # Check if the source directory exists
    if not os.path.exists(src):
        print(f"Source folder {src} does not exist.")
        return

    # Copy the folder, keeping existing contents in destination if it exists
    if os.path.exists(dst):
        print(f"Destination folder {dst} already exists. Retaining its contents.")
    else:
        print(f"Destination folder {dst} does not exist. Creating it.")
    
    # Copy contents from source to destination
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"Folder copied from {src} to {dst}.")


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

        if isinstance(answers, str) and not re.search(r"^\s*\d+\.\s+", answers, re.MULTILINE):
            # Extract type from the answers string
            type_match = re.search(r"^[^\n]+", answers)
            if type_match:
                extracted_type = type_match.group(0).strip()
                parsed_answers = {0: extracted_type}
            else:
                parsed_answers = {0: answers.strip()}
        else:
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


def generate_answers_for_fine_tuning(json_data, file_path):
    # Read and parse the JSON file

    repo_data = [entry for entry in json_data if entry.get("file") == file_path]
    counter = 1
    answers = []
    for fact in repo_data:
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


# def get_token_count(text, prompt_id=None):

#     # Ensure text is a string
#     if isinstance(text, list):
#         # Convert list of dictionaries to list of strings
#         text = " ".join(
#             [str(item) if isinstance(item, dict) else item for item in text]
#         )

#     encoding = tiktoken.encoding_for_model("gpt-4o")
#     number_of_tokens_4o = len(encoding.encode(text))

#     return number_of_tokens_4o


def get_token_count(text, prompt_id=None):
    """
    Calculate the token count for the input text.
    Supports strings, lists of strings, and lists of dictionaries with 'content' keys.
    """
    # Load the tokenizer
    encoding = tiktoken.encoding_for_model("gpt-4o")

    if isinstance(text, list):
        # If text is a list of dictionaries or strings
        if all(isinstance(item, dict) and "content" in item for item in text):
            # Extract 'content' field from dictionaries
            text = " ".join(item["content"] for item in text)
        else:
            # Join list of strings
            text = " ".join(str(item) for item in text)

    # Encode and count tokens
    tokens = encoding.encode(text)
    token_count = len(tokens)

    return token_count


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


def truncate_prompt(
    prompt, token_limit, tokenizer=tiktoken.encoding_for_model("gpt-4o")
):
    """
    Truncate the prompt to fit within the specified token limit using a tokenizer.
    :param prompt: The original prompt (list of dictionaries).
    :param token_limit: The maximum number of tokens allowed.
    :param tokenizer: The tokenizer to count tokens.
    :return: The truncated prompt.
    """
    total_tokens = 0
    truncated_prompt = []

    for message in prompt:
        # Tokenize the content of the message
        try:
            message_tokens = tokenizer.encode(
                message["content"]
            )  # Adjust tokenizer usage as needed
        except TypeError:
            # Fallback if the tokenizer doesn't have encode
            message_tokens = tokenizer.tokenize(message["content"])

        token_count = len(message_tokens)

        if total_tokens + token_count > token_limit:
            # Calculate remaining tokens and truncate the message
            remaining_tokens = token_limit - total_tokens
            truncated_message_tokens = message_tokens[:remaining_tokens]
            truncated_message = tokenizer.decode(truncated_message_tokens)
            truncated_prompt.append(
                {"role": message["role"], "content": truncated_message}
            )
            break
        else:
            truncated_prompt.append(message)
            total_tokens += token_count

    # Recalculate token count to confirm
    # full_prompt = ''.join([msg['content'] for msg in truncated_prompt])
    # token_counts = get_token_count(truncated_prompt)
    # logger.info(f"Truncated prompt to fit within the token limit. New token count: {token_counts}")

    return truncated_prompt


# def generate_csv(token_counts, prompt_id):
#     """
#     Generates a CSV file with the token count information.

#     Args:
#         token_counts (dict): Token counts for different models.
#         prompt_id (str): Identifier for the prompt template.
#     """
#     csv_file = "prompt_token_counts.csv"
#     fieldnames = ["prompt_id", "model", "token_count"]

#     # Check if the file exists
#     file_exists = os.path.isfile(csv_file)

#     with open(csv_file, mode="a", newline="") as file:
#         writer = csv.DictWriter(file, fieldnames=fieldnames)

#         # Write the header only if the file doesn't exist
#         if not file_exists:
#             writer.writeheader()

#         for model, token_count in token_counts.items():
#             writer.writerow(
#                 {"prompt_id": prompt_id, "model": model, "token_count": token_count}
#             )


def get_prompt(
    prompt_id,
    source_code,
    metadata=None,
    answers_placeholders=True,
    use_system_prompt=True,
    file_path=None,
    token_limit=8192,
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

    if token_counts > token_limit:
        global exceeded_limit_count
        exceeded_limit_count += 1
        # Load existing data if the file exists
        if os.path.exists("exceeded_token_limit_files.json"):
            with open("exceeded_token_limit_files.json", "r") as file:
                data = json.load(file)
        else:
            data = {"file_paths": []}

        # Append the new file path
        data["file_paths"].append(file_path)

        # Write the updated data back to the file
        with open("exceeded_token_limit_files.json", "w") as file:
            json.dump(data, file, indent=4)
        return None
    else:
        global within_limit_count
        within_limit_count += 1

    # Generate CSV
    # generate_csv(token_counts, prompt_id)

    # Log the final counts
    # logger.info(
    #     f"Number of prompts that exceeded the token limit: {exceeded_limit_count}"
    # )
    # logger.info(f"Number of prompts within the token limit: {within_limit_count}")

    return prompt


def process_mapping(mapping):
    assistant_message = {
        "role": "assistant",
        "content": generate_answers_for_fine_tuning(mapping["json_data"], mapping["file_path"]),
    }
    mapping["prompt"].append(assistant_message)
    return mapping["prompt"]

def dump_ft_jsonl(id_mapping, output_file):
    # Load the first mapping's JSON file
    first_mapping = next(iter(id_mapping.values()))
    with open(first_mapping["json_filepath"], "r") as file:
        json_data = json.load(file)

    mappings = copy.deepcopy(id_mapping)
    with Pool() as pool:
        with tqdm(total=len(mappings), desc="Processing mappings") as pbar:
            prompts = []
            for mapping in mappings.values():
                mapping["json_data"] = json_data
                result = process_mapping(mapping)
                prompts.append(result)
                pbar.update()

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
