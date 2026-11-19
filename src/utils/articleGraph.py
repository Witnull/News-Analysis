import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from ..core.types import EntityNode, RelationEdge
from .constants import TYPE_COLORS, DEFAULT_COLOR


@dataclass
class ArticleGraph:
    article_url: str
    article_title: str
    article_tags: List[str]
    article_date: str
    nodes: dict = field(default_factory=dict)   # id -> EntityNode
    edges: List[RelationEdge] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def list_nodes(self, type_filter: Optional[str] = None) -> List[EntityNode]:
        """Return all nodes, optionally filtered by entity type."""
        result = list(self.nodes.values())
        if type_filter:
            result = [n for n in result if n.type.lower() == type_filter.lower()]
        return sorted(result, key=lambda n: (-n.confidence, n.label))

    def list_edges(
        self,
        relation_filter: Optional[str] = None,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
    ) -> List[RelationEdge]:
        """
        Return deduplicated edges, optionally filtered by:
            - relation_filter : exact relation label  (e.g. "published_in")
            - source_id       : edges leaving this node id
            - target_id       : edges arriving at this node id
        """
        seen: set = set()
        result: List[RelationEdge] = []
        for edge in self.edges:
            key = (edge.source_id, edge.relation, edge.target_id)
            if key in seen:
                continue
            seen.add(key)
            if relation_filter and edge.relation != relation_filter:
                continue
            if source_id and edge.source_id != source_id:
                continue
            if target_id and edge.target_id != target_id:
                continue
            result.append(edge)
        return sorted(result, key=lambda e: (-e.confidence, e.relation))

    def summary(self) -> str:
        """Summary: node counts by type + edge counts by relation."""
        from collections import Counter
        type_counts = Counter(n.type for n in self.nodes.values())
        node_str = ", ".join(f"{t}: {c}" for t, c in sorted(type_counts.items()))

        unique_edges = {(e.source_id, e.relation, e.target_id) for e in self.edges}
        rel_counts = Counter(rel for _, rel, _ in unique_edges)
        edge_str = ", ".join(f"{r}: {c}" for r, c in sorted(rel_counts.items()))

        return (
            f"{self.article_title[:60]}\n"
            f"  Nodes : {len(self.nodes)}  ({node_str})\n"
            f"  Edges : {len(unique_edges)}  ({edge_str})"
        )
    # ------------------------------------------------------------------
    # DOT export
    # ------------------------------------------------------------------

    def to_dot(self, max_label_len: int = 32) -> str:

        def _esc(s: str) -> str:
            _table = {'\\': '\\\\', '"': '\\"', '\n': '\\n'}
            return ''.join(_table.get(c, c) for c in s)

        def _wrap(s: str) -> str:
            return "\\n".join(textwrap.wrap(s, max_label_len))

        tag_str = ", ".join(self.article_tags) if self.article_tags else ""
        title_line = _esc(self.article_title) + " [" + _esc(self.article_date) + "]"
        tag_line   = "Tags: " + _esc(tag_str)

        lines_out = [
            "digraph ArticleGraph {",
            "  rankdir=LR;",
            '  label="' + title_line + r"\n" + tag_line + '";  ',
            "  labelloc=t; labeljust=l;",
            '  graph [fontname="Helvetica" fontsize=11 bgcolor="#FAFAFA"];',
            '  node  [fontname="Helvetica" fontsize=10 style=filled shape=box ',
            '         margin="0.15,0.08" penwidth=1.2];',
            '  edge  [fontname="Helvetica" fontsize=9 color="#555555"];',
            "  // url: " + _esc(self.article_url),
            "",
        ]

        for node in self.nodes.values():
            color   = TYPE_COLORS.get(node.type.lower(), DEFAULT_COLOR)
            wrapped = _wrap(node.label)
            tooltip = _esc(
                "type: " + node.type
                + " | conf: " + f"{node.confidence:.2f}"
                + " | aliases: " + ", ".join(node.aliases[:5])
            )
            lines_out.append(
                '  "' + _esc(node.id) + '" '
                + '[label="' + _esc(wrapped) + '" '
                + 'fillcolor="' + color + '" '
                + 'tooltip="' + tooltip + '"];' 
            )

        lines_out.append("")
        seen: set = set()
        for edge in self.edges:
            key = (edge.source_id, edge.relation, edge.target_id)
            if key in seen:
                continue
            seen.add(key)
            style = "" if edge.via == "gliner2" else ' style="dashed"'
            rel   = _esc(edge.relation.replace("_", " "))
            lines_out.append(
                '  "' + _esc(edge.source_id) + '" -> "' + _esc(edge.target_id) + '" '
                + '[label="' + rel + '"]' + style + ";"
            )

        lines_out.append("}")
        return "\n".join(lines_out)

    def save_dot(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_dot(), encoding="utf-8")
        return p

