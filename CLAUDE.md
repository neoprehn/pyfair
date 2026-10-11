**Pflicht: Knowledge-Graph zuerst.** Für jede Code-Exploration in diesem
Workspace (Funktionen/Klassen/Routen finden, Aufrufketten, Architektur- und
Auswirkungsfragen) MUSST du zuerst die `codebase-memory-mcp`-Tools nutzen
(`search_graph`, `trace_path`, `get_code_snippet`, `query_graph`,
`get_architecture`, `search_code`) — nicht Grep/Read als ersten Reflex.
Grep/Read/Glob bleiben für Text, Configs, Nicht-Code-Dateien und als Ergänzung,
wenn der Graph keine Antwort liefert. Projektname beim Tool-Aufruf immer
mitgeben: `fair` (Rechenkern), `pyfair-cam` (CAM), `fair-web` (Frontend).

Lies bitte:

    Als allererstes: git pull (bzw. mind. git fetch + Abgleich mit origin/main) — es gibt seit
    2026-07-12 zwei Arbeitsgeräte (Heim-Rechner + mobiler Laptop für Zugfahrten), der lokale Stand
    kann also hinter dem sein, was vom jeweils anderen Gerät gepusht wurde.

    Nach dem Pull den Graph-Index auffrischen: index_repository für das aktuelle
    Repo aufrufen (der Pull kann Änderungen vom anderen Gerät gebracht haben,
    die der lokale Index noch nicht kennt), danach index_status prüfen (siehe
    Pflicht-Regel oben zur Graph-Nutzung selbst).

    Dieses Repo (pyfair, der Rechenkern) hat anders als fair-web/pyfair-cam keine
    eigene roadmap.md/ROADMAP-ARCHIV.md — Stand und offene Punkte ergeben sich aus
    der Git-Historie und RELEASE_NOTES.rst. Das persistente Memory (MEMORY.md +
    memory/*.md) wird in jede neue Session geladen.

    Subagent in .claude/agents/ (wird automatisch beim Session-Start erkannt, kein
    manuelles Laden nötig): testing (pytest für pyfair als reine Lib). Bei
    Code-Änderungen vor jedem Commit/Push aufrufen.

Was steht als nächstes an?
