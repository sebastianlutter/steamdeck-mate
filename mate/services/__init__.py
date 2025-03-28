import abc
import asyncio
import logging
import pkgutil
import importlib
import inspect
from typing import Dict, Any, List, Tuple, Optional, Type

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

    @abc.abstractmethod
    async def check_availability(self) -> bool:
        pass

    @abc.abstractmethod
    def config_str(self) -> str:
        pass


class ServiceDiscovery:
    """
    Manages a set of services. Periodically checks each service's status
    (available/unavailable). Allows retrieval of the 'best' service for a
    given type, e.g., 'TTS', 'STT', or 'LLM'.
    """

    def __init__(
        self,
        service_definitions: Optional[List[Tuple[Type[BaseService], str, int]]] = None
    ) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if service_definitions is None:
            service_definitions = []

        # 1) Auto-discover classes from the specified packages
        auto_discovered = self._discover_services(
            packages=["mate.services.llm", "mate.services.stt", "mate.services.tts"]
        )

        # 2) Combine user-supplied definitions with auto-discovered ones
        self.service_definitions: List[Tuple[Type[BaseService], str, int]] = (
            service_definitions + auto_discovered
        )

        # Map of service-name -> dict with:
        #   {
        #       "instance": BaseService or None,
        #       "available": bool
        #   }
        self.services: Dict[str, Dict[str, Any]] = {}

        # Lock for thread-safe read/write of self.services
        self._services_lock = asyncio.Lock()

        # Background task reference for availability checks
        self._update_task: Optional[asyncio.Task] = None

        # Allows graceful shutdown
        self._stop_event = asyncio.Event()

    def _discover_services(
        self, packages: List[str]
    ) -> List[Tuple[Type[BaseService], str, int]]:
        """
        Use pkgutil + importlib + inspect to import all modules under the given
        packages, and find all classes that inherit from BaseService (but are not
        BaseService itself).

        Returns a list of (service_class, name, priority).
        """
        discovered: List[Tuple[Type[BaseService], str, int]] = []

        for package_name in packages:
            try:
                package = importlib.import_module(package_name)
            except ImportError as e:
                self.logger.warning("Could not import package '%s': %s", package_name, e)
                continue

            if not hasattr(package, "__path__"):
                continue

            for _, mod_name, _ in pkgutil.walk_packages(
                package.__path__, package_name + "."
            ):
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
        self._stop_event.clear()
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
        Background task that updates the availability of all services
        every 3 seconds.
        """
        while not self._stop_event.is_set():
            await self._check_services_once()
            try:
                await asyncio.sleep(3)
            except asyncio.CancelledError:
                break

    async def _check_services_once(self) -> None:
        """
        Run one iteration of checks across all known service definitions.
        """
        for service_class, name, priority in self.service_definitions:
            self.logger.debug(
                "Checking service_class=%s, name=%s, priority=%s",
                service_class,
                name,
                priority,
            )
            async with self._services_lock:
                if name not in self.services:
                    try:
                        instance = service_class()
                        is_available = await instance.check_availability()
                        self.services[name] = {
                            "instance": instance,
                            "available": is_available,
                        }
                    except Exception as e:
                        self.logger.exception("Error creating/checking service: %s", e)
                        self.services[name] = {
                            "instance": None,
                            "available": False,
                        }
                else:
                    instance = self.services[name]["instance"]
                    if instance is not None:
                        try:
                            is_available = await instance.check_availability()
                            self.services[name]["available"] = is_available
                        except Exception as e:
                            self.logger.exception("Error checking service: %s", e)
                            self.services[name]["available"] = False

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
                    srv_data["instance"].service_type
                    if srv_data["instance"] else "N/A",
                    srv_data["instance"].priority
                    if srv_data["instance"] else "N/A",
                    srv_data["available"],
                )
                for srv_name, srv_data in self.services.items()
            ]
            for name, srv_type, priority, available in data:
                lines.append(
                    f"{name:<25}{srv_type:<8}{priority:<10}{str(available)}"
                )
        self.logger.info("\n".join(lines))

    async def get_best_service(self, service_type: str) -> Optional[BaseService]:
        """
        Get the best available service for the given type, i.e. the one
        that is 'available' and has the highest priority (or whichever logic
        you prefer).
        Returns None if no such service is available.
        """
        async def gather_candidates() -> List[BaseService]:
            async with self._services_lock:
                candidates: List[BaseService] = []
                for srv_data in self.services.values():
                    inst = srv_data["instance"]
                    if inst is not None and inst.service_type == service_type and srv_data["available"]:
                        candidates.append(inst)
                return candidates

        candidates = await gather_candidates()

        if not candidates:
            return None

        candidates.sort(key=lambda x: x.priority, reverse=True)
        return candidates[0]
