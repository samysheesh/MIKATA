"""Serveur MCP de test (stdio, JSON-RPC ligne par ligne), lancé en sous-processus par
tests/test_mcp.py. Deux outils : `heure` (lecture seule) et `ajouter_note` (écrit)."""

from __future__ import annotations

import json
import os
import sys


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


TOOLS = [
    {
        "name": "heure",
        "description": "Donne l'heure du serveur de test.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "ajouter_note",
        "description": "Ajoute une note.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "texte": {"type": "string", "description": "Texte de la note"},
                "urgent": {"type": "boolean"},
            },
            "required": ["texte"],
        },
    },
]


def main() -> None:
    sys.stdin.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  (UTF-8 : le protocole)
    print("journal du serveur de test", file=sys.stderr, flush=True)
    if os.environ.get("FLOOD_STDERR"):
        # Bien plus que le tampon d'un tube (64 Ko), et une ligne plus longue que la limite
        # du client : si MIKATA cessait de lire, le serveur resterait bloqué ici.
        for i in range(3000):
            print(f"ligne de journal {i} " + "x" * 100, file=sys.stderr)
        print("y" * (5 * 1024 * 1024), file=sys.stderr, flush=True)
    sys.stdout.write("ligne parasite qui n'est pas du JSON\n")
    sys.stdout.flush()
    for line in sys.stdin:
        message = json.loads(line)
        method, ident = message.get("method"), message.get("id")
        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": ident,
                    "result": {
                        "protocolVersion": message["params"]["protocolVersion"],
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "faux", "version": "1"},
                    },
                }
            )
            # Le serveur essaie d'emprunter le modèle de MIKATA : refusé.
            send({"jsonrpc": "2.0", "id": "s1", "method": "sampling/createMessage", "params": {}})
        elif method == "tools/list":
            if not message["params"].get("cursor"):
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": ident,
                        "result": {"tools": TOOLS[:1], "nextCursor": "p2"},
                    }
                )
            else:
                send({"jsonrpc": "2.0", "id": ident, "result": {"tools": TOOLS[1:]}})
        elif method == "tools/call":
            name = message["params"]["name"]
            args = message["params"].get("arguments") or {}
            if name == "planter":
                os._exit(3)  # plantage en pleine session
            if name == "heure":
                text, error = (
                    f"Il est 12 h (secret : {os.environ.get('CLE_TEST', 'aucun')}).",
                    False,
                )
            elif args.get("texte") == "échec":
                text, error = "Note refusée.", True
            else:
                text, error = (
                    f"Note ajoutée : {args.get('texte')} (urgent={args.get('urgent')}).",
                    False,
                )
            send(
                {
                    "jsonrpc": "2.0",
                    "id": ident,
                    "result": {"content": [{"type": "text", "text": text}], "isError": error},
                }
            )
        elif "id" in message and method is None:
            # Réponse de MIKATA à la demande d'échantillonnage : gardée pour le test.
            sys.stderr.write(f"réponse reçue : {json.dumps(message)}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
