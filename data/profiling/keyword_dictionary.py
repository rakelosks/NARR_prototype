"""
Multilingual keyword dictionary for template matching.

Maps English canonical terms to translations across supported languages,
with domain signals and column role hints. Enables matching against
non-English column names (Icelandic, German, French, Spanish, Dutch,
Danish, Norwegian, Finnish).

Three match modes:
  - exact:    token must appear as standalone word after splitting
  - stem:     translation appears as substring (minimum 4 characters)
  - contains: substring match (used sparingly)
"""

import re
import logging
import unicodedata
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _strip_accents(text: str) -> str:
    """Strip diacritical marks (accents) from text.

    Converts e.g. 'ár' → 'ar', 'fjöldi' → 'fjoldi', 'año' → 'ano'.
    This is essential for matching non-ASCII column names against their
    accented dictionary translations.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

_CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])")


def _column_name_words(col_name: str) -> set[str]:
    """
    Split a column name into lowercase word tokens.
    Handles _, -, ., /, (), whitespace, and camelCase.
    Also strips accents so 'ar' matches 'ár', etc.
    """
    expanded = _CAMEL_SPLIT.sub("_", col_name)
    tokens = re.split(r"[_\-.\s()/]+", expanded.lower())
    return {_strip_accents(t) for t in tokens if t}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class KeywordEntry(BaseModel):
    """A single keyword entry in the dictionary."""
    canonical: str
    translations: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Language code (ISO 639-1) → list of translations",
    )
    match_mode: str = "exact"  # exact, stem, contains
    domains: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)


class ColumnSignal(BaseModel):
    """Result of resolving a column name through the keyword dictionary."""
    column_name: str
    matched_canonicals: list[str] = Field(default_factory=list)
    domain_signals: list[str] = Field(default_factory=list)
    role_hints: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Dictionary entries
# ---------------------------------------------------------------------------

KEYWORD_DICTIONARY: list[KeywordEntry] = [
    # -----------------------------------------------------------------------
    # Shared / structural terms
    # -----------------------------------------------------------------------
    KeywordEntry(
        canonical="year",
        translations={
            "is": ["ár"],
            "de": ["jahr"],
            "fr": ["année", "annee"],
            "es": ["año", "ano"],
            "nl": ["jaar"],
            "da": ["år"],
            "no": ["år"],
            "fi": ["vuosi"],
        },
        match_mode="exact",
        domains=[],
        roles=["time_axis"],
    ),
    KeywordEntry(
        canonical="month",
        translations={
            "is": ["mánuður", "manuður", "manudur"],
            "de": ["monat"],
            "fr": ["mois"],
            "es": ["mes"],
            "nl": ["maand"],
            "da": ["måned", "maned"],
            "no": ["måned", "maned"],
            "fi": ["kuukausi"],
        },
        match_mode="exact",
        domains=[],
        roles=["time_axis"],
    ),
    KeywordEntry(
        canonical="date",
        translations={
            "is": ["dagsetning", "dags"],
            "de": ["datum"],
            "fr": ["date"],
            "es": ["fecha"],
            "nl": ["datum"],
            "da": ["dato"],
            "no": ["dato"],
            "fi": ["päivämäärä", "paivamaara", "päiväys", "pvm"],
        },
        match_mode="exact",
        domains=[],
        roles=["time_axis"],
    ),
    KeywordEntry(
        canonical="day",
        translations={
            "is": ["dagur"],
            "de": ["tag"],
            "fr": ["jour"],
            "es": ["día", "dia"],
            "nl": ["dag"],
            "da": ["dag"],
            "no": ["dag"],
            "fi": ["päivä", "paiva"],
        },
        match_mode="exact",
        domains=[],
        roles=["time_axis"],
    ),
    KeywordEntry(
        canonical="district",
        translations={
            "is": ["hverfi"],
            "de": ["bezirk"],
            "fr": ["quartier"],
            "es": ["distrito"],
            "nl": ["wijk"],
            "da": ["bydel"],
            "no": ["bydel"],
            "fi": ["kaupunginosa", "peruspiiri", "suurpiiri"],
        },
        match_mode="exact",
        domains=[],
        roles=["area"],
    ),
    KeywordEntry(
        canonical="area",
        translations={
            "is": ["svæði"],
            "de": ["gebiet"],
            "fr": ["zone"],
            "es": ["zona", "área", "area"],
            "nl": ["gebied"],
            "da": ["område", "omrade"],
            "no": ["område", "omrade"],
            "fi": ["alue"],
        },
        match_mode="exact",
        domains=[],
        roles=["area"],
    ),
    KeywordEntry(
        canonical="name",
        translations={
            "is": ["nafn", "heiti"],
            "de": ["name"],
            "fr": ["nom"],
            "es": ["nombre"],
            "nl": ["naam"],
            "da": ["navn"],
            "no": ["navn"],
            "fi": ["nimi"],
        },
        match_mode="exact",
        domains=[],
        roles=["name"],
    ),
    KeywordEntry(
        canonical="type",
        translations={
            "is": ["tegund", "flokkur"],
            "de": ["typ", "art"],
            "fr": ["type"],
            "es": ["tipo"],
            "nl": ["type"],
            "da": ["type"],
            "no": ["type"],
            "fi": ["tyyppi", "laji"],
        },
        match_mode="exact",
        domains=[],
        roles=["category"],
    ),
    KeywordEntry(
        canonical="status",
        translations={
            "is": ["staða"],
            "de": ["status"],
            "fr": ["statut"],
            "es": ["estado"],
            "nl": ["status"],
            "da": ["status"],
            "no": ["status"],
            "fi": ["tila"],
        },
        match_mode="exact",
        domains=[],
        roles=["status"],
    ),
    KeywordEntry(
        canonical="latitude",
        translations={
            "is": ["breiddargráða", "breiddargrada"],
            "de": ["breitengrad"],
            "fr": ["latitude"],
            "es": ["latitud"],
            "nl": ["breedtegraad"],
            "da": ["breddegrad"],
            "no": ["breddegrad"],
            "fi": ["leveysaste", "leveyspiiri"],
        },
        match_mode="exact",
        domains=[],
        roles=["latitude"],
    ),
    KeywordEntry(
        canonical="longitude",
        translations={
            "is": ["lengdargráða", "lengdargrada"],
            "de": ["längengrad", "langengrad"],
            "fr": ["longitude"],
            "es": ["longitud"],
            "nl": ["lengtegraad"],
            "da": ["længdegrad", "laengdegrad"],
            "no": ["lengdegrad"],
            "fi": ["pituusaste", "pituuspiiri"],
        },
        match_mode="exact",
        domains=[],
        roles=["longitude"],
    ),
    KeywordEntry(
        canonical="count",
        translations={
            "is": ["fjöldi"],
            "de": ["anzahl"],
            "fr": ["nombre", "compte"],
            "es": ["cantidad", "conteo"],
            "nl": ["aantal"],
            "da": ["antal"],
            "no": ["antall"],
            "fi": ["lukumäärä", "lukumaara", "määrä", "maara"],
        },
        match_mode="exact",
        domains=[],
        roles=["measure"],
    ),
    KeywordEntry(
        canonical="total",
        translations={
            "is": ["samtals", "heild"],
            "de": ["gesamt"],
            "fr": ["total"],
            "es": ["total"],
            "nl": ["totaal"],
            "da": ["total"],
            "no": ["totalt"],
            "fi": ["yhteensä", "yhteensa", "kokonais"],
        },
        match_mode="exact",
        domains=[],
        roles=["measure"],
    ),
    KeywordEntry(
        canonical="amount",
        translations={
            "is": ["upphæð", "upphaed"],
            "de": ["betrag"],
            "fr": ["montant"],
            "es": ["monto", "importe"],
            "nl": ["bedrag"],
            "da": ["beløb", "belob"],
            "no": ["beløp", "belop"],
            "fi": ["summa", "määrä", "maara"],
        },
        match_mode="exact",
        domains=[],
        roles=["measure"],
    ),
    KeywordEntry(
        canonical="value",
        translations={
            "is": ["gildi"],
            "de": ["wert"],
            "fr": ["valeur"],
            "es": ["valor"],
            "nl": ["waarde"],
            "da": ["værdi", "vaerdi"],
            "no": ["verdi"],
            "fi": ["arvo"],
        },
        match_mode="exact",
        domains=[],
        roles=["measure"],
    ),

    # -----------------------------------------------------------------------
    # Budget / financial domain
    # -----------------------------------------------------------------------
    KeywordEntry(
        canonical="budget",
        translations={
            "is": ["fjárhagsáætlun", "fjarhagsaaetlun"],
            "de": ["haushalt"],
            "fr": ["budget"],
            "es": ["presupuesto"],
            "nl": ["begroting"],
            "da": ["budget"],
            "no": ["budsjett"],
            "fi": ["budjetti", "talousarvio"],
        },
        match_mode="exact",
        domains=["budget"],
        roles=["financial_measure"],
    ),
    KeywordEntry(
        canonical="expenditure",
        translations={
            "is": ["útgjöld", "utgjold"],
            "de": ["ausgabe", "ausgaben"],
            "fr": ["dépense", "depense"],
            "es": ["gasto"],
            "nl": ["uitgave", "uitgaven"],
            "da": ["udgift"],
            "no": ["utgift"],
            "fi": ["meno", "menot"],
        },
        match_mode="stem",
        domains=["budget"],
        roles=["actual_spending"],
    ),
    KeywordEntry(
        canonical="revenue",
        translations={
            "is": ["tekjur"],
            "de": ["einnahme", "einnahmen"],
            "fr": ["recette"],
            "es": ["ingreso"],
            "nl": ["inkomsten"],
            "da": ["indtægt", "indtaegt"],
            "no": ["inntekt"],
            "fi": ["tulo", "tulot"],
        },
        match_mode="stem",
        domains=["budget"],
        roles=["financial_measure"],
    ),
    KeywordEntry(
        canonical="spending",
        translations={
            "is": ["eyðsla", "eydsla"],
            "de": ["ausgaben"],
            "fr": ["dépenses", "depenses"],
            "es": ["gastos"],
            "nl": ["besteding"],
            "da": ["forbrug"],
            "no": ["forbruk"],
            "fi": ["kulutus", "menot"],
        },
        match_mode="stem",
        domains=["budget"],
        roles=["financial_measure"],
    ),
    KeywordEntry(
        canonical="allocation",
        translations={
            "is": ["úthlutun", "uthlutun"],
            "de": ["zuteilung"],
            "fr": ["allocation"],
            "es": ["asignación", "asignacion"],
            "nl": ["toewijzing"],
            "da": ["tildeling"],
            "no": ["tildeling"],
            "fi": ["määräraha", "maararaha"],
        },
        match_mode="stem",
        domains=["budget"],
        roles=["financial_measure"],
    ),
    KeywordEntry(
        canonical="fiscal",
        translations={
            "is": ["reikningsár", "reikningsar"],
            "de": ["fiskal"],
            "fr": ["fiscal"],
            "es": ["fiscal"],
            "nl": ["fiscaal"],
            "da": ["fiskal"],
            "no": ["fiskal"],
            "fi": ["tilikausi"],
        },
        match_mode="stem",
        domains=["budget"],
        roles=["time_axis"],
    ),
    KeywordEntry(
        canonical="cost",
        translations={
            "is": ["kostnaður", "kostnadur"],
            "de": ["kosten"],
            "fr": ["coût", "cout"],
            "es": ["costo", "coste"],
            "nl": ["kosten"],
            "da": ["omkostning"],
            "no": ["kostnad"],
            "fi": ["kustannus", "hinta"],
        },
        match_mode="stem",
        domains=["budget"],
        roles=["financial_measure"],
    ),
    KeywordEntry(
        canonical="income",
        translations={
            "is": ["tekjur"],
            "de": ["einkommen"],
            "fr": ["revenu"],
            "es": ["ingreso"],
            "nl": ["inkomen"],
            "da": ["indkomst"],
            "no": ["inntekt"],
            "fi": ["tulo", "tulot"],
        },
        match_mode="stem",
        domains=["budget"],
        roles=["financial_measure"],
    ),
    KeywordEntry(
        canonical="grant",
        translations={
            "is": ["styrkur"],
            "de": ["zuschuss"],
            "fr": ["subvention"],
            "es": ["subvención", "subvencion"],
            "nl": ["subsidie"],
            "da": ["tilskud"],
            "no": ["tilskudd"],
            "fi": ["avustus"],
        },
        match_mode="exact",
        domains=["budget"],
        roles=["financial_measure"],
    ),
    KeywordEntry(
        canonical="tax",
        translations={
            "is": ["skattur"],
            "de": ["steuer"],
            "fr": ["impôt", "impot", "taxe"],
            "es": ["impuesto"],
            "nl": ["belasting"],
            "da": ["skat"],
            "no": ["skatt"],
            "fi": ["vero"],
        },
        match_mode="exact",
        domains=["budget"],
        roles=["financial_measure"],
    ),

    # -----------------------------------------------------------------------
    # Environmental monitoring domain
    # -----------------------------------------------------------------------
    KeywordEntry(
        canonical="air_quality",
        translations={
            "is": ["loftgæði", "loftgaedi"],
            "de": ["luftqualität", "luftqualitat"],
            "fr": ["qualité air", "qualite air"],
            "es": ["calidad aire"],
            "fi": ["ilmanlaatu"],
        },
        match_mode="stem",
        domains=["environmental"],
        roles=["env_measure"],
    ),
    KeywordEntry(
        canonical="pm2_5",
        translations={
            "is": ["svifryk"],
            "de": ["feinstaub"],
            "fi": ["pienhiukkaset"],
        },
        match_mode="contains",
        domains=["environmental"],
        roles=["env_measure"],
    ),
    KeywordEntry(
        canonical="pm10",
        translations={},
        match_mode="contains",
        domains=["environmental"],
        roles=["env_measure"],
    ),
    KeywordEntry(
        canonical="no2",
        translations={},
        match_mode="contains",
        domains=["environmental"],
        roles=["env_measure"],
    ),
    KeywordEntry(
        canonical="co2",
        translations={},
        match_mode="contains",
        domains=["environmental"],
        roles=["env_measure"],
    ),
    KeywordEntry(
        canonical="ozone",
        translations={
            "is": ["óson", "oson"],
            "de": ["ozon"],
            "fr": ["ozone"],
            "es": ["ozono"],
            "fi": ["otsoni"],
        },
        match_mode="stem",
        domains=["environmental"],
        roles=["env_measure"],
    ),
    KeywordEntry(
        canonical="temperature",
        translations={
            "is": ["hitastig", "hiti"],
            "de": ["temperatur"],
            "fr": ["température", "temperature"],
            "es": ["temperatura"],
            "nl": ["temperatuur"],
            "da": ["temperatur"],
            "no": ["temperatur"],
            "fi": ["lämpötila", "lampotila"],
        },
        match_mode="stem",
        domains=["environmental"],
        roles=["env_measure"],
    ),
    KeywordEntry(
        canonical="humidity",
        translations={
            "is": ["rakastig", "raki"],
            "de": ["feuchtigkeit", "luftfeuchtigkeit"],
            "fr": ["humidité", "humidite"],
            "es": ["humedad"],
            "nl": ["vochtigheid"],
            "da": ["fugtighed"],
            "no": ["fuktighet"],
            "fi": ["kosteus", "ilmankosteus"],
        },
        match_mode="stem",
        domains=["environmental"],
        roles=["env_measure"],
    ),
    KeywordEntry(
        canonical="noise",
        translations={
            "is": ["hávaði", "havadi"],
            "de": ["lärm", "larm"],
            "fr": ["bruit"],
            "es": ["ruido"],
            "nl": ["geluid"],
            "da": ["støj", "stoj"],
            "no": ["støy", "stoy"],
            "fi": ["melu"],
        },
        match_mode="exact",
        domains=["environmental"],
        roles=["env_measure"],
    ),
    KeywordEntry(
        canonical="emission",
        translations={
            "is": ["losun"],
            "de": ["emission"],
            "fr": ["émission", "emission"],
            "es": ["emisión", "emision"],
            "nl": ["emissie"],
            "da": ["emission"],
            "no": ["utslipp"],
            "fi": ["päästö", "paasto"],
        },
        match_mode="stem",
        domains=["environmental"],
        roles=["env_measure"],
    ),
    KeywordEntry(
        canonical="pollution",
        translations={
            "is": ["mengun"],
            "de": ["verschmutzung"],
            "fr": ["pollution"],
            "es": ["contaminación", "contaminacion"],
            "nl": ["vervuiling"],
            "da": ["forurening"],
            "no": ["forurensning"],
            "fi": ["saaste", "pilaantuminen"],
        },
        match_mode="stem",
        domains=["environmental"],
        roles=["env_measure"],
    ),
    KeywordEntry(
        canonical="sensor",
        translations={
            "is": ["skynjari", "mælir", "maelir"],
            "de": ["sensor"],
            "fr": ["capteur"],
            "es": ["sensor"],
            "nl": ["sensor"],
            "da": ["sensor"],
            "no": ["sensor"],
            "fi": ["anturi", "sensori"],
        },
        match_mode="exact",
        domains=["environmental"],
        roles=["station"],
    ),
    KeywordEntry(
        canonical="concentration",
        translations={
            "is": ["styrkur"],
            "de": ["konzentration"],
            "fr": ["concentration"],
            "es": ["concentración", "concentracion"],
            "nl": ["concentratie"],
            "da": ["koncentration"],
            "no": ["konsentrasjon"],
            "fi": ["pitoisuus"],
        },
        match_mode="exact",
        domains=["environmental"],
        roles=["env_measure"],
    ),
    KeywordEntry(
        canonical="substance",
        translations={
            "is": ["efni"],
            "de": ["stoff", "substanz"],
            "fr": ["substance"],
            "es": ["sustancia"],
            "nl": ["stof"],
            "da": ["stof"],
            "no": ["stoff"],
            "fi": ["aine"],
        },
        match_mode="exact",
        domains=["environmental"],
        roles=["category"],
    ),
    KeywordEntry(
        canonical="exceedance",
        translations={
            "is": ["yfir"],
            "de": ["überschreitung", "uberschreitung"],
            "fr": ["dépassement", "depassement"],
            "es": ["excedencia"],
            "nl": ["overschrijding"],
            "da": ["overskridelse"],
            "no": ["overskridelse"],
            "fi": ["ylitys"],
        },
        match_mode="exact",
        domains=["environmental"],
        roles=["env_measure"],
    ),
    KeywordEntry(
        canonical="measurement",
        translations={
            "is": ["mæling", "maeling", "mælingar", "maelingar"],
            "de": ["messung"],
            "fr": ["mesure"],
            "es": ["medición", "medicion"],
            "nl": ["meting"],
            "da": ["måling", "maling"],
            "no": ["måling", "maling"],
            "fi": ["mittaus"],
        },
        match_mode="stem",
        domains=["environmental"],
        roles=["env_measure"],
    ),

    # -----------------------------------------------------------------------
    # Transport / mobility domain
    # -----------------------------------------------------------------------
    KeywordEntry(
        canonical="traffic",
        translations={
            "is": ["umferð", "umferd"],
            "de": ["verkehr"],
            "fr": ["trafic", "circulation"],
            "es": ["tráfico", "trafico"],
            "nl": ["verkeer"],
            "da": ["trafik"],
            "no": ["trafikk"],
            "fi": ["liikenne"],
        },
        match_mode="stem",
        domains=["transport"],
        roles=["traffic_measure"],
    ),
    KeywordEntry(
        canonical="vehicle",
        translations={
            "is": ["ökutæki", "okutaeki", "bifreið", "bifreid"],
            "de": ["fahrzeug"],
            "fr": ["véhicule", "vehicule"],
            "es": ["vehículo", "vehiculo"],
            "nl": ["voertuig"],
            "da": ["køretøj", "koretoj"],
            "no": ["kjøretøy", "kjoretoy"],
            "fi": ["ajoneuvo"],
        },
        match_mode="stem",
        domains=["transport"],
        roles=["traffic_measure"],
    ),
    KeywordEntry(
        canonical="transit",
        translations={
            "is": ["almenningssamgöngur", "strætó", "straeto"],
            "de": ["nahverkehr"],
            "fr": ["transport en commun"],
            "es": ["transporte público", "transporte publico"],
            "nl": ["openbaar vervoer"],
            "da": ["offentlig transport"],
            "no": ["kollektivtransport"],
            "fi": ["joukkoliikenne"],
        },
        match_mode="stem",
        domains=["transport"],
        roles=["traffic_measure"],
    ),
    KeywordEntry(
        canonical="passenger",
        translations={
            "is": ["farþegi", "farthegi"],
            "de": ["fahrgast", "passagier"],
            "fr": ["passager"],
            "es": ["pasajero"],
            "nl": ["passagier"],
            "da": ["passager"],
            "no": ["passasjer"],
            "fi": ["matkustaja"],
        },
        match_mode="stem",
        domains=["transport"],
        roles=["traffic_measure"],
    ),
    KeywordEntry(
        canonical="ridership",
        translations={
            "is": ["farþegafjöldi", "farthegafjoldi"],
            "de": ["fahrgastzahl"],
            "fr": ["fréquentation", "frequentation"],
            "es": ["afluencia"],
            "fi": ["matkustajamäärä", "matkustajamaara"],
        },
        match_mode="stem",
        domains=["transport"],
        roles=["traffic_measure"],
    ),
    KeywordEntry(
        canonical="bike",
        translations={
            "is": ["hjól", "hjol", "reiðhjól", "reidhjol"],
            "de": ["fahrrad", "rad"],
            "fr": ["vélo", "velo"],
            "es": ["bicicleta"],
            "nl": ["fiets"],
            "da": ["cykel"],
            "no": ["sykkel"],
            "fi": ["pyörä", "pyora", "polkupyörä"],
        },
        match_mode="exact",
        domains=["transport"],
        roles=["mode"],
    ),
    KeywordEntry(
        canonical="bicycle",
        translations={
            "is": ["reiðhjól", "reidhjol"],
            "de": ["fahrrad"],
            "fr": ["bicyclette"],
            "es": ["bicicleta"],
            "fi": ["polkupyörä", "polkupyora"],
        },
        match_mode="stem",
        domains=["transport"],
        roles=["mode"],
    ),
    KeywordEntry(
        canonical="parking",
        translations={
            "is": ["bílastæði", "bilastaedi"],
            "de": ["parkplatz", "parken"],
            "fr": ["stationnement", "parking"],
            "es": ["aparcamiento", "estacionamiento"],
            "nl": ["parkeren"],
            "da": ["parkering"],
            "no": ["parkering"],
            "fi": ["pysäköinti", "pysakointi"],
        },
        match_mode="stem",
        domains=["transport"],
        roles=["traffic_measure"],
    ),
    KeywordEntry(
        canonical="speed",
        translations={
            "is": ["hraði", "hradi"],
            "de": ["geschwindigkeit"],
            "fr": ["vitesse"],
            "es": ["velocidad"],
            "nl": ["snelheid"],
            "da": ["hastighed"],
            "no": ["hastighet"],
            "fi": ["nopeus"],
        },
        match_mode="exact",
        domains=["transport"],
        roles=["traffic_measure"],
    ),
    KeywordEntry(
        canonical="flow",
        translations={
            "is": ["flæði", "flaedi"],
            "de": ["fluss", "verkehrsfluss"],
            "fr": ["flux"],
            "es": ["flujo"],
            "nl": ["stroom"],
            "da": ["flow"],
            "no": ["flyt"],
            "fi": ["virta", "liikennevirta"],
        },
        match_mode="exact",
        domains=["transport"],
        roles=["traffic_measure"],
    ),
    KeywordEntry(
        canonical="counter",
        translations={
            "is": ["teljari"],
            "de": ["zähler", "zahler"],
            "fr": ["compteur"],
            "es": ["contador"],
            "nl": ["teller"],
            "da": ["tæller", "taeller"],
            "no": ["teller"],
            "fi": ["laskuri", "laskentapiste"],
        },
        match_mode="exact",
        domains=["transport"],
        roles=["station"],
    ),
    KeywordEntry(
        canonical="bus",
        translations={
            "is": ["strætó", "straeto", "strætisvagn"],
            "de": ["bus"],
            "fr": ["bus", "autobus"],
            "es": ["autobús", "autobus"],
            "nl": ["bus"],
            "da": ["bus"],
            "no": ["buss"],
            "fi": ["bussi", "linja-auto"],
        },
        match_mode="exact",
        domains=["transport"],
        roles=["mode"],
    ),
    KeywordEntry(
        canonical="train",
        translations={
            "is": ["lest"],
            "de": ["zug", "bahn"],
            "fr": ["train"],
            "es": ["tren"],
            "nl": ["trein"],
            "da": ["tog"],
            "no": ["tog"],
            "fi": ["juna"],
        },
        match_mode="exact",
        domains=["transport"],
        roles=["mode"],
    ),

    # -----------------------------------------------------------------------
    # Demographic domain
    # -----------------------------------------------------------------------
    KeywordEntry(
        canonical="population",
        translations={
            "is": ["íbúafjöldi", "ibuafjoldi", "íbúar", "ibuar"],
            "de": ["bevölkerung", "bevolkerung", "einwohner"],
            "fr": ["population"],
            "es": ["población", "poblacion"],
            "nl": ["bevolking"],
            "da": ["befolkning"],
            "no": ["befolkning"],
            "fi": ["väestö", "vaesto", "asukasluku"],
        },
        match_mode="stem",
        domains=["demographic"],
        roles=["population_measure"],
    ),
    KeywordEntry(
        canonical="age",
        translations={
            "is": ["aldur"],
            "de": ["alter"],
            "fr": ["âge", "age"],
            "es": ["edad"],
            "nl": ["leeftijd"],
            "da": ["alder"],
            "no": ["alder"],
            "fi": ["ikä", "ika"],
        },
        match_mode="exact",
        domains=["demographic"],
        roles=["demographic_group"],
    ),
    KeywordEntry(
        canonical="gender",
        translations={
            "is": ["kyn"],
            "de": ["geschlecht"],
            "fr": ["sexe", "genre"],
            "es": ["género", "genero", "sexo"],
            "nl": ["geslacht"],
            "da": ["køn", "kon"],
            "no": ["kjønn", "kjonn"],
            "fi": ["sukupuoli"],
        },
        match_mode="exact",
        domains=["demographic"],
        roles=["demographic_group"],
    ),
    KeywordEntry(
        canonical="household",
        translations={
            "is": ["heimili", "heimila"],
            "de": ["haushalt"],
            "fr": ["ménage", "menage"],
            "es": ["hogar"],
            "nl": ["huishouden"],
            "da": ["husstand"],
            "no": ["husholdning"],
            "fi": ["kotitalous", "asuntokunta"],
        },
        match_mode="stem",
        domains=["demographic"],
        roles=["demographic_group"],
    ),
    KeywordEntry(
        canonical="census",
        translations={
            "is": ["manntal"],
            "de": ["volkszählung", "volkszahlung", "zensus"],
            "fr": ["recensement"],
            "es": ["censo"],
            "nl": ["volkstelling"],
            "da": ["folketælling", "folketaelling"],
            "no": ["folketelling"],
            "fi": ["väestölaskenta", "vaestolaskenta"],
        },
        match_mode="stem",
        domains=["demographic"],
        roles=[],
    ),
    KeywordEntry(
        canonical="inhabitants",
        translations={
            "is": ["íbúar", "ibuar"],
            "de": ["einwohner"],
            "fr": ["habitants"],
            "es": ["habitantes"],
            "nl": ["inwoners"],
            "da": ["indbyggere"],
            "no": ["innbyggere"],
            "fi": ["asukkaat", "asukas"],
        },
        match_mode="stem",
        domains=["demographic"],
        roles=["population_measure"],
    ),
    KeywordEntry(
        canonical="residents",
        translations={
            "is": ["íbúar", "ibuar"],
            "de": ["bewohner"],
            "fr": ["résidents", "residents"],
            "es": ["residentes"],
            "nl": ["bewoners"],
            "da": ["beboere"],
            "no": ["beboere"],
            "fi": ["asukkaat"],
        },
        match_mode="stem",
        domains=["demographic"],
        roles=["population_measure"],
    ),
    KeywordEntry(
        canonical="density",
        translations={
            "is": ["þéttleiki", "thettleiki"],
            "de": ["dichte"],
            "fr": ["densité", "densite"],
            "es": ["densidad"],
            "nl": ["dichtheid"],
            "da": ["tæthed", "taethed"],
            "no": ["tetthet"],
            "fi": ["tiheys", "asukastiheys"],
        },
        match_mode="stem",
        domains=["demographic"],
        roles=["population_measure"],
    ),
    KeywordEntry(
        canonical="birth",
        translations={
            "is": ["fæðing", "faeding"],
            "de": ["geburt"],
            "fr": ["naissance"],
            "es": ["nacimiento"],
            "nl": ["geboorte"],
            "da": ["fødsel", "fodsel"],
            "no": ["fødsel", "fodsel"],
            "fi": ["syntymä", "syntyma"],
        },
        match_mode="stem",
        domains=["demographic"],
        roles=["population_measure"],
    ),
    KeywordEntry(
        canonical="death",
        translations={
            "is": ["dauði", "daudi", "andlát", "andlat"],
            "de": ["tod", "sterbefall"],
            "fr": ["décès", "deces"],
            "es": ["defunción", "defuncion", "muerte"],
            "nl": ["overlijden"],
            "da": ["dødsfald", "dodsfald"],
            "no": ["dødsfall", "dodsfall"],
            "fi": ["kuolema"],
        },
        match_mode="stem",
        domains=["demographic"],
        roles=["population_measure"],
    ),
    KeywordEntry(
        canonical="migration",
        translations={
            "is": ["fólksflutn", "folksflutning", "aðflutningur", "adflutningur"],
            "de": ["migration", "zuwanderung"],
            "fr": ["migration"],
            "es": ["migración", "migracion"],
            "nl": ["migratie"],
            "da": ["indvandring"],
            "no": ["innvandring"],
            "fi": ["muutto", "maahanmuutto"],
        },
        match_mode="stem",
        domains=["demographic"],
        roles=["population_measure"],
    ),

    # -----------------------------------------------------------------------
    # Facility / infrastructure domain
    # -----------------------------------------------------------------------
    KeywordEntry(
        canonical="school",
        translations={
            "is": ["skóli", "skoli"],
            "de": ["schule"],
            "fr": ["école", "ecole"],
            "es": ["escuela", "colegio"],
            "nl": ["school"],
            "da": ["skole"],
            "no": ["skole"],
            "fi": ["koulu"],
        },
        match_mode="exact",
        domains=["facility"],
        roles=["facility_type"],
    ),
    # Education-related terms
    KeywordEntry(
        canonical="students",
        translations={
            "is": ["nemendur", "nemendi", "nemenda", "nemandi", "námsmenn", "namsmenn"],
            "de": ["schüler", "schuler", "studenten"],
            "fr": ["élèves", "eleves", "étudiants", "etudiants"],
            "es": ["estudiantes", "alumnos"],
            "nl": ["studenten", "leerlingen"],
            "da": ["elever", "studerende"],
            "no": ["elever", "studenter"],
            "fi": ["opiskelijat", "oppilaat"],
        },
        match_mode="stem",
        domains=["education"],
        roles=["population_measure"],
    ),
    KeywordEntry(
        canonical="teachers",
        translations={
            "is": ["kennarar", "kennari", "kennara"],
            "de": ["lehrer"],
            "fr": ["enseignants", "professeurs"],
            "es": ["profesores", "maestros"],
            "nl": ["docenten", "leraren"],
            "da": ["lærere", "laerere"],
            "no": ["lærere", "laerere"],
            "fi": ["opettajat"],
        },
        match_mode="stem",
        domains=["education"],
        roles=["measure"],
    ),
    KeywordEntry(
        canonical="class_section",
        translations={
            "is": ["bekkjadeild", "bekkur", "deild"],
            "de": ["klasse", "abteilung"],
            "fr": ["classe", "section"],
            "es": ["clase", "sección", "seccion"],
            "nl": ["klas"],
            "da": ["klasse"],
            "no": ["klasse"],
            "fi": ["luokka"],
        },
        match_mode="stem",
        domains=["education"],
        roles=["category"],
    ),
    KeywordEntry(
        canonical="average",
        translations={
            "is": ["meðal", "medal", "meðaltal", "medaltal", "meðalfjöldi", "medalfjoldi"],
            "de": ["durchschnitt", "mittel"],
            "fr": ["moyenne"],
            "es": ["promedio", "media"],
            "nl": ["gemiddelde"],
            "da": ["gennemsnit"],
            "no": ["gjennomsnitt"],
            "fi": ["keskiarvo"],
        },
        match_mode="stem",
        domains=[],
        roles=[],
    ),
    KeywordEntry(
        canonical="hospital",
        translations={
            "is": ["sjúkrahús", "sjukrahus"],
            "de": ["krankenhaus"],
            "fr": ["hôpital", "hopital"],
            "es": ["hospital"],
            "nl": ["ziekenhuis"],
            "da": ["hospital", "sygehus"],
            "no": ["sykehus"],
            "fi": ["sairaala"],
        },
        match_mode="stem",
        domains=["facility"],
        roles=["facility_type"],
    ),
    KeywordEntry(
        canonical="library",
        translations={
            "is": ["bókasafn", "bokasafn"],
            "de": ["bibliothek"],
            "fr": ["bibliothèque", "bibliotheque"],
            "es": ["biblioteca"],
            "nl": ["bibliotheek"],
            "da": ["bibliotek"],
            "no": ["bibliotek"],
            "fi": ["kirjasto"],
        },
        match_mode="stem",
        domains=["facility"],
        roles=["facility_type"],
    ),
    KeywordEntry(
        canonical="park",
        translations={
            "is": ["garður", "gardur", "almenningsgarður"],
            "de": ["park"],
            "fr": ["parc"],
            "es": ["parque"],
            "nl": ["park"],
            "da": ["park"],
            "no": ["park"],
            "fi": ["puisto"],
        },
        match_mode="exact",
        domains=["facility"],
        roles=["facility_type"],
    ),
    KeywordEntry(
        canonical="station",
        translations={
            "is": ["stöð", "stod"],
            "de": ["station", "bahnhof"],
            "fr": ["station", "gare"],
            "es": ["estación", "estacion"],
            "nl": ["station"],
            "da": ["station"],
            "no": ["stasjon"],
            "fi": ["asema"],
        },
        match_mode="exact",
        domains=["facility"],
        roles=["facility_type"],
    ),
    KeywordEntry(
        canonical="facility",
        translations={
            "is": ["aðstaða", "adstada"],
            "de": ["einrichtung", "anlage"],
            "fr": ["établissement", "etablissement"],
            "es": ["instalación", "instalacion"],
            "nl": ["faciliteit"],
            "da": ["facilitet"],
            "no": ["anlegg"],
            "fi": ["laitos", "toimitila"],
        },
        match_mode="stem",
        domains=["facility"],
        roles=["facility_type"],
    ),
    KeywordEntry(
        canonical="building",
        translations={
            "is": ["bygging", "húsnæði", "husnaedi"],
            "de": ["gebäude", "gebaude"],
            "fr": ["bâtiment", "batiment"],
            "es": ["edificio"],
            "nl": ["gebouw"],
            "da": ["bygning"],
            "no": ["bygning"],
            "fi": ["rakennus"],
        },
        match_mode="stem",
        domains=["facility"],
        roles=["facility_type"],
    ),
    KeywordEntry(
        canonical="capacity",
        translations={
            "is": ["rými", "rymi", "afkastageta"],
            "de": ["kapazität", "kapazitat"],
            "fr": ["capacité", "capacite"],
            "es": ["capacidad"],
            "nl": ["capaciteit"],
            "da": ["kapacitet"],
            "no": ["kapasitet"],
            "fi": ["kapasiteetti"],
        },
        match_mode="stem",
        domains=["facility"],
        roles=["capacity"],
    ),
    KeywordEntry(
        canonical="shelter",
        translations={
            "is": ["skýli", "skyli"],
            "de": ["unterkunft"],
            "fr": ["abri", "refuge"],
            "es": ["refugio", "albergue"],
            "nl": ["opvang"],
            "da": ["herberg"],
            "no": ["tilfluktsrom"],
            "fi": ["suoja", "turvakoti"],
        },
        match_mode="exact",
        domains=["facility"],
        roles=["facility_type"],
    ),
    KeywordEntry(
        canonical="playground",
        translations={
            "is": ["leikvöllur", "leikvollur"],
            "de": ["spielplatz"],
            "fr": ["aire de jeux"],
            "es": ["parque infantil"],
            "nl": ["speeltuin"],
            "da": ["legeplads"],
            "no": ["lekeplass"],
            "fi": ["leikkipuisto", "leikkikenttä", "leikkikentta"],
        },
        match_mode="stem",
        domains=["facility"],
        roles=["facility_type"],
    ),

    # -----------------------------------------------------------------------
    # Incident / event domain
    # -----------------------------------------------------------------------
    KeywordEntry(
        canonical="incident",
        translations={
            "is": ["atvik"],
            "de": ["vorfall", "vorkomm"],
            "fr": ["incident"],
            "es": ["incidente"],
            "nl": ["incident"],
            "da": ["hændelse", "haendelse"],
            "no": ["hendelse"],
            "fi": ["tapahtuma", "tapaus"],
        },
        match_mode="exact",
        domains=["incident"],
        roles=["event_type"],
    ),
    KeywordEntry(
        canonical="accident",
        translations={
            "is": ["slys"],
            "de": ["unfall"],
            "fr": ["accident"],
            "es": ["accidente"],
            "nl": ["ongeluk", "ongeval"],
            "da": ["ulykke"],
            "no": ["ulykke"],
            "fi": ["onnettomuus"],
        },
        match_mode="exact",
        domains=["incident"],
        roles=["event_type"],
    ),
    KeywordEntry(
        canonical="collision",
        translations={
            "is": [
                "árekstur", "arekstur", "árekstrar", "arekstrar",
                "árekst", "arekst",
            ],
            "en": ["collision", "collisions", "crash", "crashes"],
        },
        match_mode="stem",
        domains=["incident"],
        roles=["measure", "severity"],
    ),
    KeywordEntry(
        canonical="traffic_accident",
        translations={
            "is": ["umferðarslys", "umferdarslys"],
            "en": ["traffic accident", "traffic accidents"],
        },
        match_mode="stem",
        domains=["incident"],
        roles=["measure", "severity"],
    ),
    KeywordEntry(
        canonical="casualty",
        translations={
            "is": [
                "banaslys", "slasadi", "slasaðir", "slasadir",
                "slasað", "slasad", "meiðsl", "meidsl",
            ],
            "en": ["casualt", "fatalit", "injur", "death toll"],
        },
        match_mode="stem",
        domains=["incident"],
        roles=["measure", "severity"],
    ),
    KeywordEntry(
        canonical="crime",
        translations={
            "is": ["glæpur", "glaepur", "afbrot"],
            "de": ["verbrechen", "kriminalität"],
            "fr": ["crime", "délit"],
            "es": ["crimen", "delito"],
            "nl": ["misdrijf"],
            "da": ["kriminalitet"],
            "no": ["kriminalitet"],
            "fi": ["rikos"],
        },
        match_mode="exact",
        domains=["incident"],
        roles=["event_type"],
    ),
    KeywordEntry(
        canonical="complaint",
        translations={
            "is": ["kvörtun", "kvortun", "ábending"],
            "de": ["beschwerde"],
            "fr": ["plainte"],
            "es": ["queja", "denuncia"],
            "nl": ["klacht"],
            "da": ["klage"],
            "no": ["klage"],
            "fi": ["valitus"],
        },
        match_mode="exact",
        domains=["incident"],
        roles=["event_type"],
    ),
    KeywordEntry(
        canonical="report",
        translations={
            "is": ["tilkynning", "skýrsla", "skyrsla"],
            "de": ["meldung", "bericht"],
            "fr": ["signalement"],
            "es": ["reporte", "denuncia"],
            "nl": ["melding"],
            "da": ["anmeldelse"],
            "no": ["anmeldelse"],
            "fi": ["ilmoitus"],
        },
        match_mode="exact",
        domains=["incident"],
        roles=["event_type"],
    ),
    KeywordEntry(
        canonical="violation",
        translations={
            "is": ["brot"],
            "de": ["verstoß", "verstoss"],
            "fr": ["infraction"],
            "es": ["infracción", "infraccion"],
            "nl": ["overtreding"],
            "da": ["overtrædelse", "overtraedelse"],
            "no": ["overtredelse"],
            "fi": ["rikkomus"],
        },
        match_mode="exact",
        domains=["incident"],
        roles=["event_type"],
    ),
    KeywordEntry(
        canonical="inspection",
        translations={
            "is": ["skoðun", "skodun", "eftirlitsferð"],
            "de": ["inspektion", "kontrolle"],
            "fr": ["inspection", "contrôle"],
            "es": ["inspección", "inspeccion"],
            "nl": ["inspectie"],
            "da": ["inspektion"],
            "no": ["inspeksjon"],
            "fi": ["tarkastus"],
        },
        match_mode="exact",
        domains=["incident"],
        roles=["event_type"],
    ),
    KeywordEntry(
        canonical="ticket",
        translations={
            "is": ["sekt", "miði"],
            "de": ["strafzettel", "ticket"],
            "fr": ["contravention", "amende"],
            "es": ["multa"],
            "nl": ["boete"],
            "da": ["bøde", "bode"],
            "no": ["bot"],
            "fi": ["sakko"],
        },
        match_mode="exact",
        domains=["incident"],
        roles=["event_type"],
    ),
    KeywordEntry(
        canonical="emergency",
        translations={
            "is": ["neyðartilfelli", "neydartilfelli"],
            "de": ["notfall"],
            "fr": ["urgence"],
            "es": ["emergencia"],
            "nl": ["noodgeval"],
            "da": ["nødsituation", "nodsituation"],
            "no": ["nødsituasjon", "nodsituasjon"],
            "fi": ["hätätilanne", "hatatilanne"],
        },
        match_mode="exact",
        domains=["incident"],
        roles=["event_type"],
    ),
    KeywordEntry(
        canonical="fire",
        translations={
            "is": ["eldur", "bruni"],
            "de": ["brand", "feuer"],
            "fr": ["incendie"],
            "es": ["incendio"],
            "nl": ["brand"],
            "da": ["brand"],
            "no": ["brann"],
            "fi": ["tulipalo", "palo"],
        },
        match_mode="exact",
        domains=["incident"],
        roles=["event_type"],
    ),

    # -----------------------------------------------------------------------
    # Housing / permits domain
    # -----------------------------------------------------------------------
    KeywordEntry(
        canonical="housing",
        translations={
            "is": ["húsnæði", "husnaedi", "íbúðir", "ibudir"],
            "de": ["wohnung", "wohnungsbau"],
            "fr": ["logement"],
            "es": ["vivienda"],
            "nl": ["woning", "huisvesting"],
            "da": ["bolig"],
            "no": ["bolig"],
            "fi": ["asuminen", "asunto"],
        },
        match_mode="stem",
        domains=["housing"],
        roles=["housing_type"],
    ),
    KeywordEntry(
        canonical="permit",
        translations={
            "is": ["leyfi", "byggingarleyfi"],
            "de": ["genehmigung", "baugenehmigung"],
            "fr": ["permis"],
            "es": ["permiso", "licencia"],
            "nl": ["vergunning"],
            "da": ["tilladelse"],
            "no": ["tillatelse"],
            "fi": ["lupa", "rakennuslupa"],
        },
        match_mode="exact",
        domains=["housing"],
        roles=["permit_type"],
    ),
    KeywordEntry(
        canonical="construction",
        translations={
            "is": ["bygging", "framkvæmd", "framkvamd"],
            "de": ["bau", "bauwesen"],
            "fr": ["construction"],
            "es": ["construcción", "construccion"],
            "nl": ["bouw"],
            "da": ["byggeri"],
            "no": ["bygg"],
            "fi": ["rakentaminen", "rakennustyö"],
        },
        match_mode="stem",
        domains=["housing"],
        roles=["housing_type"],
    ),
    KeywordEntry(
        canonical="rent",
        translations={
            "is": ["leiga", "húsaleiga"],
            "de": ["miete"],
            "fr": ["loyer"],
            "es": ["alquiler", "renta"],
            "nl": ["huur"],
            "da": ["leje"],
            "no": ["leie"],
            "fi": ["vuokra"],
        },
        match_mode="exact",
        domains=["housing"],
        roles=["housing_type"],
    ),
    KeywordEntry(
        canonical="property",
        translations={
            "is": ["eign", "fasteign"],
            "de": ["immobilie", "grundstück", "grundstuck"],
            "fr": ["propriété", "propriete", "bien"],
            "es": ["propiedad", "inmueble"],
            "nl": ["eigendom"],
            "da": ["ejendom"],
            "no": ["eiendom"],
            "fi": ["kiinteistö", "kiinteisto"],
        },
        match_mode="stem",
        domains=["housing"],
        roles=["housing_type"],
    ),
    KeywordEntry(
        canonical="dwelling",
        translations={
            "is": ["íbúð", "ibud"],
            "de": ["wohnung"],
            "fr": ["habitation"],
            "es": ["vivienda"],
            "nl": ["woning"],
            "da": ["bolig"],
            "no": ["bolig"],
            "fi": ["asunto"],
        },
        match_mode="stem",
        domains=["housing"],
        roles=["housing_type"],
    ),
    KeywordEntry(
        canonical="zoning",
        translations={
            "is": ["skipulag", "deiliskipulag"],
            "de": ["flächennutzung", "flachennutzung", "bebauungsplan"],
            "fr": ["zonage"],
            "es": ["zonificación", "zonificacion"],
            "nl": ["bestemmingsplan"],
            "da": ["zonering"],
            "no": ["regulering"],
            "fi": ["kaavoitus"],
        },
        match_mode="stem",
        domains=["housing"],
        roles=["housing_type"],
    ),
    KeywordEntry(
        canonical="renovation",
        translations={
            "is": ["endurbætur", "endurbaetur"],
            "de": ["renovierung", "sanierung"],
            "fr": ["rénovation", "renovation"],
            "es": ["renovación", "renovacion"],
            "nl": ["renovatie"],
            "da": ["renovering"],
            "no": ["renovering"],
            "fi": ["remontti", "peruskorjaus"],
        },
        match_mode="stem",
        domains=["housing"],
        roles=["housing_type"],
    ),
    KeywordEntry(
        canonical="planning",
        translations={
            "is": ["skipulagsmál", "skipulagsmal"],
            "de": ["planung"],
            "fr": ["urbanisme", "planification"],
            "es": ["planificación", "planificacion"],
            "nl": ["planning"],
            "da": ["planlægning", "planlaegning"],
            "no": ["planlegging"],
            "fi": ["kaavoitus", "suunnittelu"],
        },
        match_mode="stem",
        domains=["housing"],
        roles=["housing_type"],
    ),
]


# ---------------------------------------------------------------------------
# Build lookup indices for fast access
# ---------------------------------------------------------------------------

# Map canonical → entry for O(1) lookup
_CANONICAL_INDEX: dict[str, KeywordEntry] = {
    entry.canonical: entry for entry in KEYWORD_DICTIONARY
}

# Map domain → list of canonical terms in that domain
_DOMAIN_CANONICALS: dict[str, list[str]] = {}
for _entry in KEYWORD_DICTIONARY:
    for _domain in _entry.domains:
        _DOMAIN_CANONICALS.setdefault(_domain, []).append(_entry.canonical)


# ---------------------------------------------------------------------------
# WHO thresholds for environmental template
# ---------------------------------------------------------------------------

WHO_THRESHOLDS: dict[str, dict] = {
    "pm2_5": {"annual": 5, "24h": 15, "unit": "µg/m³"},
    "pm10": {"annual": 15, "24h": 45, "unit": "µg/m³"},
    "no2": {"annual": 10, "24h": 25, "unit": "µg/m³"},
    "ozone": {"peak_season": 60, "8h": 100, "unit": "µg/m³"},
    "co2": {"outdoor": 450, "unit": "ppm"},
}


def get_who_threshold(canonical: str) -> Optional[dict]:
    """Look up WHO guideline thresholds for a pollutant canonical name."""
    return WHO_THRESHOLDS.get(canonical)


# ---------------------------------------------------------------------------
# Lookup algorithm
# ---------------------------------------------------------------------------

def column_suggests_air_quality_context(col_name: str) -> bool:
    """
    True when a column name looks like a pollutant concentration field
    (µg/m³, etc.), not a financial grant column.

    Icelandic *styrkur* means both "grant" and "concentration"; this
    disambiguates using units and air-quality tokens.
    """
    if not col_name:
        return False
    n = _strip_accents(col_name.lower())
    for u in ("\u00b5", "\u03bc", "µ"):
        n = n.replace(u, "u")
    n = n.replace("m³", "m3").replace("m^3", "m3")
    markers = (
        "ug/", " ug/", "(ug", " ug ", "m3", "/m3",
        "pm10", "pm2", "no2", "so2", "o3", "h2s",
        "svifryk", "loft", "mengun", "maeling",
    )
    return any(m in n for m in markers)


def _disambiguate_styrkur_homonym(
    col_name: str,
    matched_canonicals: list[str],
    domain_signals: set[str],
    role_hints: list[str],
) -> None:
    """Resolve grant vs concentration when both dictionary entries hit *styrkur*."""
    if "grant" not in matched_canonicals or "concentration" not in matched_canonicals:
        return
    if column_suggests_air_quality_context(col_name):
        matched_canonicals[:] = [c for c in matched_canonicals if c != "grant"]
        domain_signals.discard("budget")
        role_hints[:] = [r for r in role_hints if r != "financial_measure"]
    else:
        matched_canonicals[:] = [c for c in matched_canonicals if c != "concentration"]
        domain_signals.discard("environmental")
        role_hints[:] = [r for r in role_hints if r != "env_measure"]


def resolve_column(col_name: str) -> ColumnSignal:
    """
    Resolve a column name through the keyword dictionary.

    For each dictionary entry, checks the English canonical against the
    token set (always), then checks all translations using the entry's
    match_mode.

    Returns a ColumnSignal with matched canonical terms, domain signals,
    and role hints.
    """
    words = _column_name_words(col_name)
    col_lower = _strip_accents(col_name.lower())

    matched_canonicals: list[str] = []
    domain_signals: set[str] = set()
    role_hints: list[str] = []

    for entry in KEYWORD_DICTIONARY:
        matched = False

        # Always check English canonical against token set
        canonical_words = set(entry.canonical.split("_"))
        if canonical_words & words:
            matched = True

        # Check translations
        if not matched:
            for _lang, translations in entry.translations.items():
                for trans in translations:
                    trans_lower = _strip_accents(trans.lower())
                    if entry.match_mode == "exact":
                        # Check against token set (accent-stripped)
                        trans_words = {_strip_accents(w) for w in re.split(r"[\s_\-]+", trans.lower())}
                        if trans_words & words:
                            matched = True
                            break
                    elif entry.match_mode == "stem":
                        # Substring match, minimum 4 characters
                        if len(trans_lower) >= 4 and trans_lower in col_lower:
                            matched = True
                            break
                    elif entry.match_mode == "contains":
                        # Simple substring
                        if trans_lower in col_lower:
                            matched = True
                            break
                if matched:
                    break

        if matched:
            matched_canonicals.append(entry.canonical)
            domain_signals.update(entry.domains)
            for role in entry.roles:
                if role not in role_hints:
                    role_hints.append(role)

    _disambiguate_styrkur_homonym(
        col_name, matched_canonicals, domain_signals, role_hints,
    )

    return ColumnSignal(
        column_name=col_name,
        matched_canonicals=matched_canonicals,
        domain_signals=list(domain_signals),
        role_hints=role_hints,
    )


def resolve_metadata_domains(
    title: str = "",
    description: str = "",
    tags: Optional[list[str]] = None,
) -> dict[str, int]:
    """
    Scan metadata text (title, description, tags) for domain keyword hits.

    Returns a dict of {domain: hit_count} for use as a tiebreaker.
    """
    text = f"{title} {description} {' '.join(tags or [])}".lower()
    words = set(re.split(r"[\s_\-.,;:]+", text))

    domain_hits: dict[str, int] = {}
    for entry in KEYWORD_DICTIONARY:
        if not entry.domains:
            continue

        matched = False
        # Check canonical
        canonical_words = set(entry.canonical.split("_"))
        if canonical_words & words:
            matched = True

        # Check translations
        if not matched:
            for _lang, translations in entry.translations.items():
                for trans in translations:
                    trans_lower = trans.lower()
                    if entry.match_mode == "exact":
                        trans_words = set(re.split(r"[\s_\-]+", trans_lower))
                        if trans_words & words:
                            matched = True
                            break
                    elif entry.match_mode in ("stem", "contains"):
                        if len(trans_lower) >= 4 and trans_lower in text:
                            matched = True
                            break
                if matched:
                    break

        if matched:
            for domain in entry.domains:
                domain_hits[domain] = domain_hits.get(domain, 0) + 1

    return domain_hits
