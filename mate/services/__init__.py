import abc
import asyncio
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

    def __init__(self, name: str, service_type: str, priority: int):
        self.name = name
        self.service_type = service_type
        self.priority = priority

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

    def __init__(self, service_definitions: List[Tuple[Any, str, int]] = None):
        """
        :param service_definitions: A list of tuples of the form:
            (service_class, name, priority)
          If not provided, it can be empty or partially populated. This list
          will be extended with all dynamically discovered classes.
        """
        if service_definitions is None:
            service_definitions = []

        # 1) Auto-discover classes from the specified packages
        auto_discovered = self._discover_services(
            packages=["mate.services.llm", "mate.services.stt", "mate.services.tts"]
        )

        # 2) Combine user-supplied definitions with auto-discovered ones
        self.service_definitions = service_definitions + auto_discovered

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

        For demonstration, we auto-generate a 'name' (from the class name)
        and a default priority. Adjust as needed.
        """
        discovered = []

        for package_name in packages:
            try:
                package = importlib.import_module(package_name)
            except ImportError as e:
                print(f"Could not import package '{package_name}': {e}")
                continue

            # Walk the package path for all submodules
            if not hasattr(package, "__path__"):
                # It's possible this is not a package but a single module
                continue

            for finder, mod_name, is_pkg in pkgutil.walk_packages(
                package.__path__, package_name + "."
            ):
                try:
                    mod = importlib.import_module(mod_name)
                except Exception as e:
                    print(f"Failed to import module {mod_name}: {e}")
                    continue

                # Inspect all classes in the module
                ignore_classes = ["LlmOllamaRemote"]
                for name, obj in inspect.getmembers(mod, inspect.isclass):
                    if name in ignore_classes:
                        continue
                    # Check if it's a subclass of BaseService but not BaseService itself
                    if issubclass(obj, BaseService) and obj is not BaseService and not inspect.isabstract(obj):
                        service_name = f"{obj.__name__}"
                        default_priority = -1
                        discovered.append((obj, service_name, default_priority))

        return discovered

    async def start(self):
        """
        Start periodic service availability checks in the background.
        """
        # Clear any prior stop event
        self._stop_event.clear()
        # Create a background task to periodically update status
        self._update_task = asyncio.create_task(self._update_loop())

    async def stop(self):
        """
        Signal the periodic update loop to stop and wait for it.
        """
        if self._update_task:
            self._stop_event.set()
            # Cancel any potential sleep immediately
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None

    async def _update_loop(self):
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

    async def _check_services_once(self):
        """
        Run one iteration of checks across all known service definitions.
        """
        for service_class, name, priority in self.service_definitions:
            #print(f"service_class={service_class}, name={name}, priority={priority}")
            async with self._services_lock:
                # 1) If service instance doesn't exist, attempt to create it
                if name not in self.services:
                    try:
                        instance = service_class()
                        is_available = await instance.check_availability()
                        self.services[name] = {
                            "instance": instance,
                            "available": is_available
                        }
                    except Exception as e:
                        print(e)
                        # If creation or check fails, mark unavailable
                        self.services[name] = {
                            "instance": None,
                            "available": False
                        }
                else:
                    # 2) If we already have the service, check it again
                    instance = self.services[name]["instance"]
                    if instance is not None:
                        try:
                            is_available = await instance.check_availability()
                            self.services[name]["available"] = is_available
                        except Exception:
                            self.services[name]["available"] = False

    async def print_status_table(self):
        """
        Print a simple table of the current status of all services.
        """
        lines = ["\nService Status:"]
        lines.append(f"{'NAME':<25}{'TYPE':<8}{'PRIORITY':<10}{'AVAILABLE'}")
        lines.append("-" * 55)

        # Acquire lock to read the latest data safely
        async with self._services_lock:
            data = [(srv_name,
                     srv_data["instance"].service_type if srv_data["instance"] else "N/A",
                     srv_data["instance"].priority if srv_data["instance"] else "N/A",
                     srv_data["available"])
                    for srv_name, srv_data in self.services.items()]

            for name, srv_type, priority, available in data:
                lines.append(
                    f"{name:<25}{srv_type:<8}{priority:<10}{str(available)}"
                )
            print("\n".join(lines))

    async def get_best_service(self, service_type: str) -> Optional[BaseService]:
        """
        Get the best available service for the given type, i.e. the one
        that is 'available' and has the highest priority (or whichever logic you prefer).
        Returns None if no such service is available.
        """
        async def gather_candidates():
            async with self._services_lock:
                candidates = []
                for srv_name, srv_data in self.services.items():
                    inst = srv_data["instance"]
                    is_available = srv_data["available"]
                    if inst is not None \
                       and inst.service_type == service_type \
                       and is_available:
                        candidates.append(inst)
                return candidates

        candidates = await gather_candidates()

        if not candidates:
            return None
        # Sort by priority descending (higher = better), or adjust logic as needed
        candidates.sort(key=lambda x: x.priority, reverse=True)
        return candidates[0]
