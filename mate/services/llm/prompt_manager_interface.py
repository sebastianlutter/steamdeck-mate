import logging
import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar


# Define Modes
class Mode(Enum):
    EXIT = """
    Wähle EXIT wenn der User das Gespräch beenden oder abbrechen will oder sich verabschieded hat.
    """
    GARBAGEINPUT = """
    Wähle GARBAGEINPUT wenn die Anfrage unverständlich oder unvollständig erscheint.
    """
    LEDCONTROL = """
    Wähle LEDCONTROL wenn der User die Beleuchtung oder das Licht verändern, ein oder ausschalten möchte.
    """
    STATUS = """
    Wähle STATUS wenn der User von Geräten (Fernseher, Verstärker) oder Dinge ein- oder ausschalten will (ausser wenn es um Licht geht).
    """
    CHAT = """
    Wähle CHAT wenn der User eine andere bisher nicht genannte Frage gestellt hat, oder sonstiger Small Talk oder verständlichen Satz ohne Bezug zu den anderen Themen. Im Zweifel diese Option wählen wenn der Input eine valide Frage darstellt.
    """
    MODUS_SELECTION = ''


@dataclass
class PromptTemplate:
    mode: Mode
    system_prompt: str
    user_say_str: str
    description: str

    def format_prompt(self, context_data: Optional[Dict[str, str]] = None) -> str:
        if context_data is None:
            context_data = {}
        system_prompt_formatted: str = self.system_prompt.format(**context_data)
        return system_prompt_formatted


H = TypeVar("H")  # History type
E = TypeVar("E")  # Entry type

GLOBAL_BASE_TEMPLATES: Dict[str, PromptTemplate] = {
    Mode.MODUS_SELECTION.name: PromptTemplate(
        mode=Mode.MODUS_SELECTION,
        description="Modus Auswahl",
        system_prompt=(
                "Du musst genau einen der folgenden Modi (GROSSBUCHSTABEN) wählen: "
                f"{', '.join([mode.name for mode in Mode if mode != Mode.MODUS_SELECTION])}\n"
                "Beginne deine Antwort, indem du den gewählten Modus in GROSSBUCHSTABEN nennst (z. B. \"EXIT\"). "
                "Beende deine Antwort danach. Keine weiteren Erklärungen, Haftungsausschlüsse oder zusätzlicher Text.\n\n"
                "Befolge diese Regeln strikt:\n"
                + "\n".join(f"- {m.value}" for m in Mode if m.value)
        ),
        user_say_str=""
    ),
    Mode.CHAT.name: PromptTemplate(
        mode=Mode.CHAT,
        description="Live Chat Modus",
        system_prompt=(
            "Beantworte die Fragen als freundlicher und zuvorkommender Helfer. "
            "Antworte kindergerecht für Kinder ab acht Jahren. "
            "Antworte maximal mit 1 bis 3 kurzen Sätzen und stelle Gegenfragen, wenn der Sachverhalt unklar ist."
        ),
        user_say_str="Lass uns etwas plaudern, Modus ist nun CHAT"
    ),
    Mode.LEDCONTROL.name: PromptTemplate(
        mode=Mode.LEDCONTROL,
        description="LED Kontroll Modus",
        system_prompt="""
Du steuerst LED-Lichter per JSON requests. 
Der User möchte sie möglicherweise ein- oder ausschalten oder die Farbe oder Helligkeit ändern. 

Parameter und mögliche Werte:
- action: on, off. Nur off wenn User Dunkelheit will oder das Licht ausgeschalten werden soll und kein Licht mehr an sein soll. Immer on wählen wenn etwas am Licht verändret werden soll.
- rgbww: Array mit fünf Elementen: Rot, Grün, Blau, kaltes Weiß, warmes Weiß (jeweils von 0 bis 255).
- colortemp: Farbtemperatur setzen (2200K bis 6500K).
- brightness: Helligkeit anpassen, dünkler oder heller (Wertebereich 10–255).
- scene von 1 bis 32 ruft vordefinierte (oft dynamische) Szenen auf.  
- scene 0 wird für benutzerdefinierte Farben oder Farbtemperaturen genutzt.  
- speed (0–100) ist nur für dynamische Szenen relevant und bestimmt die Geschwindigkeit der Farbübergänge.    
- temp oder rgbww werden nur beachtet, wenn sceneId = 0. 

Stelle sicher, dass deine endgültige Ausgabe ein kurzes JSON-Snippet im folgendem Format ist:
Der action parameter ist mandatory, andere parameter sind je nach Modus zu wählen.

Scene ID Reference (Tabelle)

| scene | Scene Name   | Beschreibung / Hinweise                                                      | Statisch oder Dynamisch? | Typischerweise relevante Parameter 
|--------------|-------------------|----------------------------------------------------------------------------------|------------------------------|----------------------------|
| 1            | Ocean            | Langsame Farbwechsel in Blau- und Grüntönen.                                      | Dynamisch                    |  speed |
| 2            | Romance          | Warme, langsame Überblendungen in Rosa-, Rot- und Violetttönen.                  | Dynamisch                    |  speed  |
| 3            | Sunset           | Tiefe Orange- und Rottöne, die einen Sonnenuntergang nachahmen.                  | Dynamisch                    |  speed  |
| 4            | Party            | Helle, lebhafte Farbwechsel.                                                     | Dynamisch                    |  speed  |
| 5            | Fireplace        | Flackernde Rot/Orange-Töne, ähnlich einem Kaminfeuer.                            | Dynamisch                    |  speed  |
| 6            | Cozy             | Warme Orange/Brauntöne in dezenter Übergangsform.                                 | Dynamisch                    |  speed |
| 7            | Forest           | Grüne und erdige Farbtöne.                                                       | Dynamisch                    |  speed  |
| 8            | Pastel Colors    | Sanfte Pastell-Farbwechsel (z.B. hellblau, rosa, hellgelb).                       | Dynamisch                    |  speed |
| 9            | Wake up          | Allmähliches Aufhellen, oft genutzt für Morgenroutinen.                          | Dynamisch                    |  speed  |
| 10           | Bedtime          | Allmähliches Abdunkeln zu wärmeren Farbtönen, für die Nacht.                     | Dynamisch                    |  speed  |
| 11           | Warm White       | Standard-Warmweiß (ca. 2700K–3000K).                                              | Statisch                     |         |
| 12           | Daylight         | Neutral- bis kaltweißes Licht (ca. 5000K–5500K).                                  | Statisch                     |         |
| 13           | Cool white       | Kälteres Weiß (ca. 6000K–6500K).                                                 | Statisch                     |          |
| 14           | Night light      | Sehr gedimmtes, warmes Licht.                                                    | Statisch                     |          |
| 15           | Focus            | Meist ein kühleres Weiß (um 6500K) und hell.                                      | Statisch                     |         |
| 16           | Relax            | Meist ein warm- bis neutralweißes Licht.                                         | Statisch                     |          |
| 17           | True colors      | Betont eine natürliche Farbwiedergabe, oft ca. 4000K.                             | Statisch                     |         |
| 18           | TV time          | Sanftes, warmes Weiß, teils leichte Farbwechsel.                                 | Überwiegend statisch         |          |
| 19           | Plant growth     | Violett-/Rosa-Töne für Pflanzenbeleuchtung.                                      | Überwiegend statisch         |          |
| 20           | Spring           | Zarte grünliche Pastell-Verläufe.                                                | Dynamisch                    |  speed  |
| 21           | Summer           | Hellere, warme Farbübergänge, die an Sonnenschein erinnern.                      | Dynamisch                    |  speed  |
| 22           | Fall             | Kräftige Rot-/Orangetöne, die an Herbstlaub erinnern.                            | Dynamisch                    |  speed  |
| 23           | Deep dive        | Tiefe Blau- und Violetttöne.                                                     | Dynamisch                    |  speed  |
| 24           | Jungle           | Grüne, ggf. dezente Übergänge.                                                   | Dynamisch                    |  speed  |
| 25           | Mojito           | Grüne und gelbe Farbwechsel.                                                     | Dynamisch                    |  speed  |
| 26           | Club             | Schnelle Farbwechsel in hellen und kräftigen Tönen.                              | Dynamisch                    |  speed  |
| 27           | Christmas        | Rot-Grün-Wechsel.                                                                | Dynamisch                    |  speed  |
| 28           | Halloween        | Orange-Lila-Wechsel.                                                             | Dynamisch                    |  speed  |
| 29           | Candlelight      | Sehr warmes Flackern.                                                            | Dynamisch                    |  speed  |
| 30           | Golden white     | Etwas wärmeres Weiß als „Warm White“.                                            | Statisch                     |          |
| 31           | Pulse            | Pulsierende, kräftige Farbzyklen.                                                | Dynamisch                    |  speed  |
| 32           | Steampunk        | Warme Bernstein-Übergänge mit leichtem „mechanischen“ Flacker-Effekt.           | Dynamisch                    |  speed   |

Einige Beispiele:
Wärmstes Licht: {'action': 'on', 'scene': 0, 'colortemp': 2200, 'brightness': 255}
Tageslicht: {'action': 'on','scene': 12, 'colortemp': 4200, 'brightness': 255}
Nachtlicht: {'action': 'on','scene': 14}
Gemütlich: {'action': 'on','scene': 6, 'brightness': 255}
Entspannung: {'action': 'on','scene': 16, 'brightness': 255}
Color light with given rgb: {'action': 'on','rgbww': [255, 0, 0, 0, 0], 'scene': 0, 'brightness': 255}
Animated Fireplace light: {'action': 'on','scene': 5, 'speed': 100, 'brightness': 255}

Beachte das rgbww ein Tupel mit 5 elementen ist.
Beachte die wichtigste Regel strikt: Antworte mit EINER EINZELNEN JSON Ausgabe die den Endzustand beschreibt, und beende danach. Keine weiteren Erklärungen, Haftungsausschlüsse oder zusätzlicher Text.
""",
        user_say_str=""
    ),
    Mode.GARBAGEINPUT.name: PromptTemplate(
        mode=Mode.GARBAGEINPUT,
        description="Unverständlicher Input",
        system_prompt=(
            "Die Benutzereingabe ist unverständlich oder unvollständig. "
            "Bitte fordere den Benutzer auf, die Anfrage zu präzisieren."
        ),
        user_say_str=""
    ),
    Mode.EXIT.name: PromptTemplate(
        mode=Mode.STATUS,
        description="Anzeigen des System Status",
        system_prompt="",
        user_say_str=""
    ),
    Mode.STATUS.name: PromptTemplate(
        mode=Mode.EXIT,
        description="Beenden",
        system_prompt="",
        user_say_str=""
    ),
}


class ReductionStrategy(ABC):
    @abstractmethod
    def reduce(self, history: List[Dict[str, str]], tokenize_fn: Any, token_limit: int) -> None:
        pass


class RemoveOldestStrategy(ReductionStrategy):
    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug("RemoveOldestStrategy initialized.")

    def reduce(self, history: List[Dict[str, str]], tokenize_fn: Any, token_limit: int) -> None:
        while self.calculate_token_count(history, tokenize_fn) > token_limit and history:
            removed_entry = history.pop(0)
            self.logger.debug("Removed entry to reduce tokens: %s", removed_entry)

    def calculate_token_count(self, history: List[Dict[str, str]], tokenize_fn: Any) -> int:
        total_tokens: int = 0
        for entry in history:
            content: str = entry.get("content", "")
            total_tokens += tokenize_fn(content)
        self.logger.debug("Calculated total tokens: %d", total_tokens)
        return total_tokens


class PromptManager(ABC, Generic[H, E]):
    def __init__(self, initial_mode: Mode, reduction_strategy: Optional[ReductionStrategy]) -> None:
        if reduction_strategy is None:
            self.reduction_strategy: ReductionStrategy = RemoveOldestStrategy()
        else:
            self.reduction_strategy = reduction_strategy

        self.current_mode: Mode = initial_mode
        self.template: PromptTemplate = GLOBAL_BASE_TEMPLATES[initial_mode.name]
        self.histories: Dict[Mode, List[Dict[str, str]]] = {mode: [] for mode in Mode}

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(
            "Initialized histories for modes: %s",
            [mode.name for mode in self.histories.keys()]
        )
        self.logger.debug("Initial mode set to %s", self.current_mode.name)

    def set_mode(self, mode: Mode) -> None:
        if mode not in self.histories:
            self.logger.error("Attempted to set unsupported mode: %s", mode.name)
            raise ValueError(f"Mode {mode.name} is not supported for history management.")
        self.current_mode = mode
        self.template = GLOBAL_BASE_TEMPLATES[mode.name]
        self.logger.info("Mode set to %s", self.current_mode.name)

    @abstractmethod
    def set_history(self, history: H) -> None:
        pass

    @abstractmethod
    def empty_history(self) -> None:
        pass

    @abstractmethod
    def get_history(self) -> H:
        pass

    @abstractmethod
    def get_last_entry(self) -> Optional[E]:
        pass

    @abstractmethod
    def add_user_entry(self, user_prompt: str) -> E:
        pass

    @abstractmethod
    def add_assistant_entry(self, ai_response: str) -> E:
        pass

    @abstractmethod
    def count_history_tokens(self) -> int:
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

    @abstractmethod
    def reduce_history(self, token_limit: int) -> None:
        pass

    @abstractmethod
    def pretty_print_history(self) -> str:
        pass

    def get_system_prompt(self, context_data: Optional[Dict[str, str]] = None) -> str:
        system_prompt: str = self.template.format_prompt(context_data)
        self.logger.debug("System prompt retrieved: %s", system_prompt)
        return system_prompt

    def get_timestamp(self) -> str:
        now: datetime.datetime = datetime.datetime.now(datetime.timezone.utc)
        return f"Es ist {now.strftime('%A')}, der {now.strftime('%d.%m.%Y')} um {now.strftime('%H:%M')} UTC. "
