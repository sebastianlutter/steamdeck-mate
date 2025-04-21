import asyncio

import yaml
from typing import Dict, Any, List, Type
import importlib

from mate.services import BaseService
from mate.services.llm.llm_interface import LlmInterface
from mate.services.stt.stt_interface import STTInterface
from mate.services.tts.tts_interface import TTSInterface
import logging
import json

logger = logging.getLogger(f"{__name__}.service_loader")

# Helper to dynamically import a class
def import_class_from_path(path: str) -> Type:
    module_path, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

# Dynamically creates a subclass with a custom class name and constructor
def create_dynamic_class(class_name: str, base_class: Type, config: Dict[str, Any]) -> Type:
    def __init__(self_self):
        #logger.debug(f"Dynamic class constructor: {config}")
        super(new_class, self_self).__init__(**config)

    new_class = type(class_name, (base_class,), {
        "__init__": __init__,
        "config": config
    })
    # return the blueprint to create the class (constructor is not called yet)
    return new_class


async def create_instances_by_key(yaml_path: str, key: str) -> List[BaseService]:
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    # Use dictionary lookup since yaml_data is a dict
    section = yaml_data.get(key, [])

    instances = []
    for entry in section:
        # Process each entry from the section to create your instance.
        class_name = entry.get("name")
        # get and remove base_class name (is not a constructor parameter)
        base_class_path = entry.pop("base_class")
        base_class = import_class_from_path(base_class_path)
        logger.debug(f"create {key} service instance: {class_name}:\n## Class: {base_class.__name__} ##\n{json.dumps(entry, indent=4, sort_keys=True)}")
        instance = create_dynamic_class(class_name, base_class, entry)
        instances.append(instance)  # Replace with actual instance creation
    return instances

# Individual service creation functions
async def create_ollama_llm_instances(yaml_path: str = "remote_services.yml") -> List[LlmInterface]:
    return await create_instances_by_key(yaml_path=yaml_path, key="LLM")

async def create_openrouter_openai_instances(yaml_path: str = "remote_services.yml") -> List[LlmInterface]:
    return await create_instances_by_key(yaml_path=yaml_path, key="LLM")

async def create_stt_instances(yaml_path: str = "remote_services.yml") -> List[STTInterface]:
    return await create_instances_by_key(yaml_path=yaml_path, key="STT")

async def create_tts_instances(yaml_path: str = "remote_services.yml") -> List[TTSInterface]:
    return await create_instances_by_key(yaml_path=yaml_path, key="TTS")

async def create_service_instances(yaml_path: str = "remote_services.yml") -> List[BaseService]:
    llm_ollama_instances, llm_openrouter_instances, stt_instance, tts_instance = await asyncio.gather(
        create_ollama_llm_instances(yaml_path=yaml_path),
        create_openrouter_openai_instances(yaml_path=yaml_path),
        create_stt_instances(yaml_path=yaml_path),
        create_tts_instances(yaml_path=yaml_path)
    )
    return llm_ollama_instances + llm_openrouter_instances + stt_instance + tts_instance