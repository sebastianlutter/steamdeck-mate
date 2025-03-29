import abc
import sys

from mate.services.llm.prompt_manager_interface import PromptManager


class BaseService(abc.ABC):
    """
    Abstract interface for all services.
    Each service should implement:
      - A service_type (string)
      - A priority (int; higher = better or lower = better, you decide)
      - An async check_availability() method
    """

    def __init__(self, name: str, service_type: str, priority: int) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.name: str = name
        self.service_type: str = service_type
        self.priority: int = priority

    async def __check_remote_endpoint__(self, endpoint: str) -> bool:
        parsed = urlparse(endpoint)
        host: Optional[str] = parsed.hostname
        port: Optional[int] = parsed.port

        if not host or not port:
            self.logger.debug(
                "[check_availability %s] Invalid endpoint: %s (missing host or port)",
                self.name,
                endpoint
            )
            return False

        # Try to open a connection with a 2-second timeout.
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=2
            )
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            self.logger.debug(
                "[check_availability %s] Could not connect to host '%s' on port %s. Reason: %s",
                self.name,
                host,
                port,
                e
            )
            return False
        return True

    @abc.abstractmethod
    async def check_availability(self) -> bool:
        pass

    @abc.abstractmethod
    def config_str(self) -> str:
        pass


import asyncio
import threading
import logging
import pkgutil
import importlib
import inspect
import re
from typing import Optional, List, Tuple, Type, Dict, Any
from urllib.parse import urlparse

# Assuming BaseService is defined elsewhere.
# from your_module import BaseService

class ServiceDiscovery:
    """
    Manages a set of services. Periodically checks each service's status
    (available/unavailable). Allows retrieval of the 'best' service for a
    given type, e.g., 'TTS', 'STT', or 'LLM'.

    Now implemented as a thread-safe Singleton with double-checked locking.
    """

    _instance: Optional["ServiceDiscovery"] = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:  # Fast path without lock.
            with cls._instance_lock:
                if cls._instance is None:  # Double-check under lock.
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        service_definitions: Optional[List[Tuple[Type["BaseService"], str, int]]] = None
    ) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if service_definitions is None:
            service_definitions = []

        # 1) Auto-discover classes from the specified packages.
        auto_discovered = self._discover_services(
            packages=["mate.services.llm", "mate.services.stt", "mate.services.tts"]
        )

        # 2) Combine user-supplied definitions with auto-discovered ones.
        self.service_definitions: List[Tuple[Type["BaseService"], str, int]] = (
            service_definitions + auto_discovered
        )

        # Map of service name -> dict with "instance" and "available"
        self.services: Dict[str, Dict[str, Any]] = {}

        # Lock for thread-safe access to self.services (used inside async tasks).
        self._services_lock = asyncio.Lock()

        # Background task reference for availability checks.
        self._update_task: Optional[asyncio.Task] = None

        # Allows graceful shutdown.
        self._stop_event = asyncio.Event()

    def _discover_services(
        self, packages: List[str]
    ) -> List[Tuple[Type["BaseService"], str, int]]:
        """
        Use pkgutil + importlib + inspect to import all modules under the given
        packages, and find all classes that inherit from BaseService (but are not
        BaseService itself).

        Returns a list of (service_class, name, priority).
        """
        discovered: List[Tuple[Type["BaseService"], str, int]] = []

        for package_name in packages:
            try:
                package = importlib.import_module(package_name)
            except ImportError as e:
                self.logger.warning("Could not import package '%s': %s", package_name, e)
                continue

            if not hasattr(package, "__path__"):
                continue

            for _, mod_name, _ in pkgutil.walk_packages(package.__path__, package_name + "."):
                try:
                    mod = importlib.import_module(mod_name)
                except Exception as e:
                    self.logger.warning("Failed to import module %s: %s", mod_name, e)
                    continue

                ignore_classes = ["LlmOllamaRemote", "STTWhisperRemote", "TTSOpenedAISpeech"]
                for name, obj in inspect.getmembers(mod, inspect.isclass):
                    if name in ignore_classes:
                        continue
                    if (
                        issubclass(obj, BaseService)
                        and obj is not BaseService
                        and not inspect.isabstract(obj)
                    ):
                        service_name = f"{obj.__name__}"
                        default_priority = -1
                        discovered.append((obj, service_name, default_priority))
        return discovered

    async def start(self) -> None:
        """
        Start periodic service availability checks in the background.
        """
        self.logger.info("Start service discovery")
        self._stop_event.clear()
        # Run one full scan and wait for it to finish.
        await self._check_services_once()
        self._update_task = asyncio.create_task(self._update_loop())

    async def stop(self) -> None:
        """
        Signal the periodic update loop to stop and wait for it.
        """
        if self._update_task:
            self._stop_event.set()
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None

    async def _update_loop(self) -> None:
        """
        Background task that updates the availability of all services every 3 seconds.
        """
        while not self._stop_event.is_set():
            await self._check_services_once()
            try:
                await asyncio.sleep(3)
            except asyncio.CancelledError:
                break

    async def _check_services_once(self) -> None:
        """
        Run one iteration of checks across all known service definitions in parallel.
        """
        async def check_one(service_class: Type["BaseService"], name: str, priority: int):
            # If an instance already exists, reuse it; otherwise, create a new one.
            async with self._services_lock:
                existing = self.services.get(name)
            if existing is not None:
                instance = existing.get("instance")
            else:
                instance = None

            if instance is None:
                try:
                    instance = service_class()
                except Exception as e:
                    self.logger.exception("Error creating service %s: %s", name, e)
                    return (name, None, False)
            try:
                is_available = await instance.check_availability()
            except Exception as e:
                self.logger.exception("Error checking service %s: %s", name, e)
                is_available = False
            return (name, instance, is_available)

        tasks = [
            asyncio.create_task(check_one(service_class, name, priority))
            for service_class, name, priority in self.service_definitions
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        async with self._services_lock:
            for name, instance, is_available in results:
                self.services[name] = {"instance": instance, "available": is_available}

    async def print_status_table(self) -> None:
        """
        Print a simple table of the current status of all services.
        """
        lines = ["\nService Status:"]
        lines.append(f"{'NAME':<25}{'TYPE':<8}{'PRIORITY':<10}{'AVAILABLE'}")
        lines.append("-" * 55)

        async with self._services_lock:
            data = [
                (
                    srv_name,
                    srv_data["instance"].service_type if srv_data["instance"] else "N/A",
                    srv_data["instance"].priority if srv_data["instance"] else "N/A",
                    srv_data["available"],
                )
                for srv_name, srv_data in self.services.items()
            ]
        for name, srv_type, priority, available in data:
            lines.append(f"{name:<25}{srv_type:<8}{priority:<10}{str(available)}")
        self.logger.info("\n".join(lines))

    async def get_best_service(self, service_type: str) -> Optional["BaseService"]:
        """
        Get the best available service for the given type, i.e. the one that is 'available'
        and has the highest priority (or whichever logic you prefer).
        Returns None if no such service is available.
        """
        async with self._services_lock:
            candidates = [
                srv_data["instance"]
                for srv_data in self.services.values()
                if srv_data["instance"] and srv_data["instance"].service_type == service_type and srv_data["available"]
            ]

        if not candidates:
            await self.print_status_table()
            print()
            sys.exit(f"There is no {service_type} service available. You may want to run \"./docker/docker.sh\" to bring up local instances.")

        candidates.sort(key=lambda x: x.priority, reverse=True)
        self.logger.info(f"type: {service_type}, return {candidates[0].name}")
        return candidates[0]
