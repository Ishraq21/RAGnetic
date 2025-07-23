import logging
import hcl2
import re
from typing import Any, Dict, List, Optional, Type
from pydantic.v1 import BaseModel, Field

logger = logging.getLogger(__name__)


class TerraformParserInput(BaseModel):
    """Input schema for the TerraformParserTool."""
    hcl_code: str = Field(..., description="HCL code from a Terraform file.")
    resource_type: str = Field(..., description="Type of resource to extract, e.g. 'aws_instance'.")


class TerraformParserTool:
    """Parses Terraform HCL to extract all resources of a given type."""
    name = "terraform_parser_tool"
    description = (
        "Given HCL code, finds all `resource \"<type>\" \"<name>\" { ... }` blocks "
        "and returns their configurations."
    )
    args_schema: Type[BaseModel] = TerraformParserInput

    def get_input_schema(self) -> Dict[str, Any]:
        return self.args_schema.schema()

    def _strip_comments_and_strings(self, code: str) -> str:
        """
        Remove comments (//, #, /* */) and mask out quoted-string contents
        so braces inside strings don’t upset our depth counting.
        """
        out = []
        i = 0
        n = len(code)
        while i < n:
            if code.startswith("//", i) or code.startswith("#", i):
                # skip to end of line
                i = code.find("\n", i)
                if i < 0:
                    break
                continue
            if code.startswith("/*", i):
                # skip C‑style comment
                j = code.find("*/", i + 2)
                i = j + 2 if j >= 0 else n
                continue
            ch = code[i]
            if ch in ('"', "'"):
                # mask out entire string literal
                quote = ch
                out.append(ch)
                i += 1
                while i < n:
                    if code[i] == "\\":
                        out.append(code[i:i+2])
                        i += 2
                        continue
                    if code[i] == quote:
                        out.append(quote)
                        i += 1
                        break
                    # replace content with space
                    out.append(" ")
                    i += 1
            else:
                out.append(ch)
                i += 1
        return "".join(out)

    def _clean_hcl(self, hcl_code: str) -> str:
        """
        Remove markdown dividers and then extract balanced HCL blocks
        by counting braces on the comment‑/string‑stripped text.
        """
        # first strip horizontal rules
        lines = [l for l in hcl_code.splitlines() if l.strip() != "---"]
        code = "\n".join(lines)

        # strip comments & mask strings for safe depth counting
        safe = self._strip_comments_and_strings(code)

        parts, buf, depth = [], [], 0
        for orig_line, safe_line in zip(code.splitlines(), safe.splitlines()):
            buf.append(orig_line)
            depth += safe_line.count("{") - safe_line.count("}")
            if depth == 0 and buf:
                parts.append("\n".join(buf))
                buf = []
        cleaned = "\n\n".join(parts).strip()

        if not cleaned:
            logger.warning("Cleaning yielded nothing—falling back to minimal strip.")
            cleaned = "\n".join(lines).strip()
        return cleaned

    def run(
        self,
        hcl_code: Optional[str] = None,
        resource_type: Optional[str] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        # 1) gather inputs
        hcl_code = hcl_code or kwargs.get("hcl_code") or kwargs.get("code", "")
        resource_type = resource_type or kwargs.get("resource_type") or kwargs.get("type", "")
        if not hcl_code:
            return [{"error": "Missing 'hcl_code' argument."}]
        if not resource_type:
            logger.warning("No resource_type given—defaulting to 'aws_instance'.")
            resource_type = "aws_instance"

        logger.info(f"Looking for resources of type '{resource_type}'…")
        # 2) snippet extraction by regex + brace‑matching
        snippet_re = re.compile(rf'^\s*resource\s+"{re.escape(resource_type)}"\s+"([^"]+)"\s*\{{')
        found: List[Dict[str, Any]] = []
        lines = hcl_code.splitlines()
        for i, line in enumerate(lines):
            m = snippet_re.match(line)
            if not m:
                continue
            name = m.group(1)
            depth = self._strip_comments_and_strings(line).count("{") - self._strip_comments_and_strings(line).count("}")
            block = [line]
            j = i + 1
            while j < len(lines) and depth > 0:
                block_line = lines[j]
                block.append(block_line)
                safe = self._strip_comments_and_strings(block_line)
                depth += safe.count("{") - safe.count("}")
                j += 1
            snippet = "\n".join(block)
            try:
                parsed = hcl2.loads(snippet)
                for res in parsed.get("resource", []):
                    cfg = res.get(resource_type, {}).get(name)
                    if cfg is not None:
                        cfg["local_name"] = name
                        found.append(cfg)
            except Exception as e:
                logger.error(f"Snippet parse failed for {resource_type} '{name}': {e}")

        # 3) fallback to full‑file clean & parse if needed
        if not found:
            logger.warning("No isolated blocks; falling back to full‑file parse…")
            cleaned = self._clean_hcl(hcl_code)
            try:
                parsed_full = hcl2.loads(cleaned)
                for res in parsed_full.get("resource", []):
                    if resource_type in res:
                        for name, cfg in res[resource_type].items():
                            cfg["local_name"] = name
                            found.append(cfg)
            except Exception as e:
                logger.error(f"Full‑file parse failed: {e}")
                return [{"error": f"Failed to parse HCL after cleaning: {e}"}]

        if not found:
            return [{"error": f"No '{resource_type}' resources found in HCL code."}]

        logger.info(f"Discovered {len(found)} '{resource_type}' resources.")
        return found
