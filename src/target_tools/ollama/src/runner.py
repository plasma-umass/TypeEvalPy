import argparse
import json
import logging
import re
import shutil
import sys
import traceback
from pathlib import Path
from sys import stdout
from typing import List, Optional

import prompts
import utils
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama, OpenAI
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel

AUTOFIX_WITH_OPENAI = False
ENABLE_STREAMING = False


class TypeEvalPySchema(BaseModel):
    file: str
    line_number: int
    type: List[str]
    function: Optional[str] = None
    parameter: Optional[str] = None
    variable: Optional[str] = None


PROMPTS_MAP = {
    "json_based_1": prompts.json_based_1,
    "json_based_2": prompts.json_based_2,
    "questions_based_1": prompts.questions_based_1,
}

# Create a logger
logger = logging.getLogger("runner")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("/tmp/ollama_log.log")
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(stdout)
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def list_python_files(folder_path):
    python_files = sorted(Path(folder_path).rglob("*.py"))
    return python_files


def get_prompt(prompt_id, code_path, json_filepath):
    # with open(json_filepath, "r") as file:
    #     data = json.load(file)
    with open(code_path, "r") as file:
        code = file.read()

    if prompt_id == "questions_based_1":
        questions_from_json = utils.generate_questions_from_json(json_filepath)

        prompt = PromptTemplate(
            template=PROMPTS_MAP[prompt_id],
            input_variables=["code", "questions"],
        )

        prompt_data = {
            "code": code,
            "questions": "\nResult:\n".join(questions_from_json),
        }
    elif prompt_id in ["json_based_1", "json_based_2"]:
        parser = PydanticOutputParser(pydantic_object=TypeEvalPySchema)

        prompt = PromptTemplate(
            template=PROMPTS_MAP[prompt_id],
            input_variables=["code", "filename"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        prompt_data = {"code": code, "filename": Path(code_path).name}
    else:
        print("ERROR! Prompt not found!")
        sys.exit(-1)

    _input = prompt.format_prompt(**prompt_data)

    return _input.to_string()


def process_file(file_path, llm, openai_llm, prompt_id):
    try:
        error_count = 0
        json_filepath = str(file_path).replace(".py", "_gt.json")
        result_filepath = str(file_path).replace(".py", f"_result.json")

        try:
            output = llm.invoke(get_prompt(prompt_id, file_path, json_filepath))

            # TODO: Include this in langchain pipeline
            output = re.sub(r"```json", "", output)
            output = re.sub(r"```", "", output)

            # response = llm.invoke(file_prompt)
            if AUTOFIX_WITH_OPENAI:
                new_parser = OutputFixingParser.from_llm(parser=parser, llm=openai_llm)
                output = new_parser.parse(output)

            logger.info(output)
            utils.generate_json_file(result_filepath, output)

        except Exception as e:
            traceback.print_exc()
            logger.info(f"{file_path} failed: {e}")
            error_count = 1

        return error_count
    except Exception as e:
        logger.info(f"{file_path} failed: {e}")
        raise


def main_runner(args):
    error_count = 0
    model_name = "text-davinci-003"
    temperature = 0.0
    openai_llm = OpenAI(
        model_name=model_name, temperature=temperature, openai_api_key=args.openai_key
    )

    for model in args.ollama_models:
        files_analyzed = 0

        # Create result folder for model specific results
        bechmark_path = Path(args.bechmark_path)
        results_src = bechmark_path
        results_dst = bechmark_path.parent / model / bechmark_path.name
        utils.copy_folder(results_src, results_dst)

        python_files = list_python_files(results_dst)

        # TODO: Add gpt as model here
        llm = Ollama(
            model=model,
            callback_manager=(
                CallbackManager([StreamingStdOutCallbackHandler()])
                if ENABLE_STREAMING
                else None
            ),
            temperature=0.0,
        )
        llm.base_url = args.ollama_url

        for file in python_files:
            try:
                logger.info(file)
                inferred = process_file(file, llm, openai_llm, args.prompt_id)

            except Exception as e:
                logger.info(
                    f"Command returned non-zero exit status: {e} for file: {file}"
                )
                error_count += 1

            files_analyzed += 1
            logger.info(
                f"Progress: {files_analyzed}/{len(python_files)} Errors: {error_count}"
            )

    logger.info(f"Runner finished with errors:{error_count}")


if __name__ == "__main__":
    print("Python is running inside a Docker container")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bechmark_path",
        help="Specify the benchmark path",
        default="/tmp/micro-benchmark",
    )

    parser.add_argument(
        "--ollama_url", help="Specify the ollama server url", required=True
    )

    parser.add_argument("--prompt_id", help="Specify the prompt ID", required=True)

    parser.add_argument(
        "--ollama_models",
        nargs="+",
        type=str,
        help="Space-separated list of ollama models",
        required=True,
    )

    parser.add_argument("--openai_key", help="Openai API key", required=True)

    args = parser.parse_args()
    main_runner(args)
