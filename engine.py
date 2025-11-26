import os
import re
import random
import glob
import math
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod

# --- Exceptions ---

class StopParsingException(Exception):
    """Raised when [stop] is encountered to halt all generation."""
    pass

# --- Base Command Interface ---

class BaseCommand(ABC):
    """Abstract base class for all grammar commands with unified argument parsing."""
    
    name: str = "base"
    is_container: bool = False

    def parse_args(self, arg_str: str) -> Tuple[List[str], Dict[str, str]]:
        """
        Parses arguments into positional args and named kwargs.
        
        Input: 'type="action" tags="social" recursive="false"'
        Output: ([], {'type': 'action', 'tags': 'social', 'recursive': 'false'})
        """
        arg_str = arg_str.strip()
        if not arg_str:
            return [], {}
        
        pattern = re.compile(r'(?:(?P<key>\w+)=)?(?P<val>"[^"]*"|\'[^\']*\'|[^\s]+)')
        
        args = []
        kwargs = {}
        
        for match in pattern.finditer(arg_str):
            k = match.group('key')
            v = match.group('val')
            
            # Strip quotes from value
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            
            if k:
                kwargs[k] = v
            else:
                args.append(v)
                
        return args, kwargs

    @abstractmethod
    def execute(self, parser: 'GrammarParser', arg: str, content: str) -> Tuple[str, bool]:
        pass

# --- Grammar Parser Core ---

class GrammarParser:
    def __init__(self, root_dir: str, seed: int):
        self.root = root_dir
        self.root_abs = os.path.abspath(root_dir) if root_dir else os.getcwd()
        
        self.rng = random.Random(seed)
        
        self.vars: Dict[str, str] = {}
        self.overrides: Dict[str, str] = {}
        self.macros: Dict[str, str] = {}
        self.intercepts: Dict[str, str] = {} 
        
        # Registry: A flat list of items. Each item is {"tags": set, "content": str}
        self.registry: List[Dict[str, Any]] = []
        
        self.vars['seed'] = str(seed)
        self.last_condition_result: Optional[bool] = None
        self.intercept_pass_triggered = False # Flag for [pass] logic
        self.max_recursion_depth = 50
        self.stop_triggered = False 
        self.break_triggered = False
        
        self.commands: Dict[str, BaseCommand] = {}
        self._register_default_commands()

    def register_command(self, command: BaseCommand):
        self.commands[command.name] = command

    def _register_default_commands(self):
        # STANDALONE
        self.register_command(CmdStop())
        self.register_command(CmdBreak())
        self.register_command(CmdPass())
        self.register_command(CmdLog())
        self.register_command(CmdGet())
        self.register_command(CmdInc())
        self.register_command(CmdDec())
        self.register_command(CmdExists())
        self.register_command(CmdFile())
        self.register_command(CmdCall())
        self.register_command(CmdSelect())
        self.register_command(CmdLibrary())
        self.register_command(CmdRange())
        self.register_command(CmdAll())

        # CONTAINER
        self.register_command(CmdSet())
        self.register_command(CmdOverride())
        self.register_command(CmdCalc())
        self.register_command(CmdIf())
        self.register_command(CmdElseIf())
        self.register_command(CmdElse())
        self.register_command(CmdSwitch())
        self.register_command(CmdLoop())
        self.register_command(CmdRegister())
        self.register_command(CmdIntercept())
        self.register_command(CmdDef())
        self.register_command(CmdRan())
        self.register_command(CmdShuffle())
        self.register_command(CmdJoin())
        self.register_command(CmdChance())
        self.register_command(CmdLen())
        self.register_command(CmdComment())
        self.register_command(CmdCommentHash())
        self.register_command(CmdIgnore())
        self.register_command(CmdMute())
        
        # WEIGHTING COMMANDS
        self.register_command(CmdRW())
        self.register_command(CmdIRW())

    def _cleanup_text(self, text: str) -> str:
        text = re.sub(r'[\r\n]+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\s+([,.])', r'\1', text)
        text = re.sub(r',+', ',', text)
        text = re.sub(r'^,', '', text)
        text = re.sub(r',(?=[a-zA-Z])', ', ', text)
        return text.strip()

    def parse(self, text: str, depth: int = 0) -> str:
        if self.stop_triggered: return ""
        if depth > self.max_recursion_depth: return text 

        i = 0
        while i < len(text):
            if self.stop_triggered:
                text = text[:i]
                break
            if self.break_triggered:
                return text

            start_tag_index = text.find('[', i)
            if start_tag_index == -1: break
            
            end_head_index = self._find_matching_bracket(text, start_tag_index)
            if end_head_index == -1:
                i = start_tag_index + 1
                continue

            head_content = text[start_tag_index + 1 : end_head_index].strip()
            if not head_content:
                i = end_head_index + 1
                continue

            parts = head_content.split(None, 1)
            tag_name = parts[0]
            tag_arg = parts[1] if len(parts) > 1 else ""

            if tag_name not in self.commands:
                i = start_tag_index + 1
                continue
            
            command = self.commands[tag_name]

            if '[' in tag_arg:
                tag_arg = self.parse(tag_arg, depth + 1)
                if self.stop_triggered or self.break_triggered:
                    return text[:start_tag_index]

            replacement = ""
            full_match_len = 0
            inner_content = ""

            if not command.is_container:
                full_match_len = (end_head_index - start_tag_index) + 1
                replacement, should_reparse = command.execute(self, tag_arg, "")
            else:
                close_tag_str = f"[/{tag_name}]"
                if tag_name == '#': close_tag_str = "[/#]"

                content_start = end_head_index + 1
                content_end = self._find_closing_tag(text, content_start, f"[{tag_name}", close_tag_str)

                if content_end == -1:
                    i = end_head_index + 1
                    continue

                inner_content = text[content_start:content_end]
                full_match_len = (content_end + len(close_tag_str)) - start_tag_index
                
                replacement, should_reparse = command.execute(self, tag_arg, inner_content)

            if should_reparse and replacement:
                replacement = self.parse(replacement, depth + 1)
                
                # Check for [pass] signal from inner content
                if self.intercept_pass_triggered and tag_name != 'select':
                    # Propagate up unless handled
                    pass 

            if self.break_triggered and tag_name != 'loop':
                 return text[:start_tag_index] + replacement

            text = text[:start_tag_index] + replacement + text[start_tag_index + full_match_len:]
            i = start_tag_index + len(replacement)

        if depth == 0:
            text = self._cleanup_text(text)

        return text

    def _find_matching_bracket(self, text: str, start_index: int) -> int:
        depth = 0
        for i in range(start_index, len(text)):
            if text[i] == '[': depth += 1
            elif text[i] == ']':
                depth -= 1
                if depth == 0: return i
        return -1

    def _find_closing_tag(self, text: str, start_index: int, open_tag_start: str, close_tag: str) -> int:
        depth = 1
        current_index = start_index
        while depth > 0:
            next_open = text.find(open_tag_start, current_index)
            next_close = text.find(close_tag, current_index)
            if next_close == -1: return -1

            valid_open = False
            if next_open != -1:
                after_open = next_open + len(open_tag_start)
                if after_open < len(text) and text[after_open] in (' ', ']'):
                    valid_open = True

            if next_open != -1 and next_open < next_close and valid_open:
                depth += 1
                current_index = next_open + len(open_tag_start)
            else:
                depth -= 1
                current_index = next_close + len(close_tag)
                if depth == 0: return next_close
        return -1

    def resolve_var(self, name: str) -> str:
        return self.overrides.get(name, self.vars.get(name, ""))
    
    def load_file_content(self, filename: str, recursive: bool = True) -> str:
        if not filename or not self.root: return ""
        filename = filename.strip()
        abs_path = os.path.abspath(os.path.join(self.root_abs, filename))
        
        if abs_path.startswith(self.root_abs) and os.path.exists(abs_path) and os.path.isdir(abs_path):
            return self.load_folder_content(filename, recursive=recursive)
            
        full_path = abs_path if abs_path.endswith('.txt') else f"{abs_path}.txt"
        
        if not full_path.startswith(self.root_abs): return ""
        if os.path.exists(full_path) and os.path.isfile(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f: return f.read()
            except: return ""
        return ""

    def load_folder_content(self, folder_name: str, recursive: bool = True) -> str:
        if not folder_name or not self.root: return ""
        target_dir = os.path.abspath(os.path.join(self.root_abs, folder_name.strip()))
        if not target_dir.startswith(self.root_abs) or not os.path.isdir(target_dir): return ""
        
        pattern = "**/*.txt" if recursive else "*.txt"
        files = glob.glob(os.path.join(target_dir, pattern), recursive=recursive)
        files.sort()
        
        results = []
        for file_path in files:
            # --- IGNORE DOT DIRECTORIES ---
            rel_path = os.path.relpath(file_path, self.root_abs)
            if any(part.startswith('.') for part in rel_path.split(os.sep)):
                continue
            
            if not os.path.isfile(file_path): continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f: results.append(f.read())
            except: continue
        return "\n".join(results)

    def split_safe(self, text: str, separator: str = '|') -> List[str]:
        parts = []
        last_split = 0
        depth = 0
        container_depth = 0
        
        i = 0
        n = len(text)
        
        while i < n:
            char = text[i]
            
            if char == '[':
                # Attempt to identify tag to handle nested containers properly
                j = i + 1
                is_close = False
                
                if j < n and text[j] == '/':
                    is_close = True
                    j += 1
                
                # Skip leading whitespace
                while j < n and text[j].isspace():
                    j += 1
                
                # Extract potential tag name
                name_start = j
                while j < n and (text[j].isalnum() or text[j] in ('_', '#', '@')):
                    j += 1
                
                tag_name = text[name_start:j]
                
                # Check if this tag is a registered container
                if tag_name in self.commands and self.commands[tag_name].is_container:
                    if is_close:
                        container_depth = max(0, container_depth - 1)
                    else:
                        container_depth += 1
                
                depth += 1
                
            elif char == ']':
                depth = max(0, depth - 1)
                
            elif char == separator and depth == 0 and container_depth == 0:
                parts.append(text[last_split:i])
                last_split = i + 1
                
            i += 1
            
        parts.append(text[last_split:])
        return [p.strip() for p in parts if p.strip()]

    def parse_list(self, content: str) -> Tuple[List[str], str]:
        if '|' in content: return [x.strip() for x in content.split('|') if x.strip()], '|'
        lines = [l for l in content.splitlines() if l.strip()]
        if len(lines) > 0: return lines, '\n'
        if content.strip(): return [content.strip()], '\n'
        return [], '\n'

    def safe_eval(self, expression: str) -> str:
        allowed = {
            "abs": abs, "round": round, "min": min, "max": max, "int": int, "float": float, "str": str,
            "math": math, "pi": math.pi, "e": math.e, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, 
            "tan": math.tan, "ceil": math.ceil, "floor": math.floor
        }
        try:
            if any(x in expression for x in ['__', 'import', 'lambda']): return ""
            result = eval(expression, {"__builtins__": None}, allowed)
            if isinstance(result, float) and result.is_integer(): return str(int(result))
            return str(result)
        except: return ""

# --- Modular Registry Commands ---

class CmdRegister(BaseCommand):
    """
    Registers content into the global registry.
    """
    def __init__(self):
        self.name = "register"
        self.is_container = True

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        args, kwargs = self.parse_args(arg)
        tags = set()
        
        def add_tags(s):
            for t in s.split(','):
                if t.strip(): tags.add(t.strip().lower())

        if args: add_tags(args[0])
        if "tags" in kwargs: add_tags(kwargs["tags"])
        if "type" in kwargs: add_tags(kwargs["type"])
        
        item = {"tags": tags, "content": content}
        parser.registry.append(item)
            
        return "", False

class CmdIntercept(BaseCommand):
    """
    Defines a specific override for a [select] call with matching required tags.
    """
    def __init__(self):
        self.name = "intercept"
        self.is_container = True

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        args, kwargs = self.parse_args(arg)
        tags = set()
        
        source = kwargs.get("tags", args[0] if args else "")
        if source:
            for t in source.split(','):
                if t.strip(): tags.add(t.strip().lower())

        if tags:
            # Create a consistent key: "a,b" and "b,a" become "a|b"
            key = "|".join(sorted(list(tags)))
            parser.intercepts[key] = content
            
        return "", False

class CmdPass(BaseCommand):
    """
    Signals the parent [select] command to skip the current intercept 
    and proceed to the next candidate or registry.
    """
    def __init__(self):
        self.name = "pass"
        self.is_container = False

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        parser.intercept_pass_triggered = True
        return "", False

class CmdLibrary(BaseCommand):
    """
    Loads and parses text files from a directory into memory.
    """
    def __init__(self):
        self.name = "library"
        self.is_container = False

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        args, kwargs = self.parse_args(arg)
        target = kwargs.get("dir", args[0] if args else "")
        
        rec_val = kwargs.get("recursive", "true").lower()
        recursive = rec_val != "false"
        
        if target:
            full_content = parser.load_folder_content(target, recursive=recursive)
            parser.parse(full_content)
        return "", False

class CmdSelect(BaseCommand):
    """
    Selects a random item from the registry matching criteria.
    """
    def __init__(self):
        self.name = "select"
        self.is_container = False

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        args, kwargs = self.parse_args(arg)
        
        req_tags = set()
        any_tags = set()
        ex_tags = set()

        def parse_to_set(source_str, target_set, allow_inline_exclude=False):
            if not source_str: return
            parts = [t.strip() for t in source_str.split(',') if t.strip()]
            for p in parts:
                if allow_inline_exclude and p.startswith(('!', '-', '~')):
                    ex_tags.add(p[1:].lower())
                else:
                    target_set.add(p.lower())

        if args: parse_to_set(args[0], req_tags, allow_inline_exclude=True)
        if "tags" in kwargs: parse_to_set(kwargs["tags"], req_tags, allow_inline_exclude=True)
        if "required" in kwargs: parse_to_set(kwargs["required"], req_tags, allow_inline_exclude=True)
        if "type" in kwargs: parse_to_set(kwargs["type"], req_tags)
        
        if "any" in kwargs: parse_to_set(kwargs["any"], any_tags)
        if "oneof" in kwargs: parse_to_set(kwargs["oneof"], any_tags)
        
        if "exclude" in kwargs: parse_to_set(kwargs["exclude"], ex_tags)

        # --- 1. Standard Registry Search ---
        candidates = []
        for item in parser.registry:
            i_tags = item["tags"]
            
            if not req_tags.issubset(i_tags): continue
            if not ex_tags.isdisjoint(i_tags): continue
            if any_tags and any_tags.isdisjoint(i_tags): continue

            candidates.append(item)

        if not candidates:
            return "", False

        # --- 2. Pick a Winner ---
        selected_item = parser.rng.choice(candidates)
        winner_tags = selected_item["tags"]

        # --- 3. Check for Intercepts on the Winner ---
        potential_intercepts = []
        for key, val in parser.intercepts.items():
            int_tags = set(key.split('|'))
            if int_tags.issubset(winner_tags):
                potential_intercepts.append((int_tags, val))
        
        # Sort specific to generic (longest matches first)
        potential_intercepts.sort(key=lambda x: len(x[0]), reverse=True)

        # --- 4. Attempt Execution ---
        for _, int_content in potential_intercepts:
            parser.intercept_pass_triggered = False
            result = parser.parse(int_content)
            
            if parser.intercept_pass_triggered:
                parser.intercept_pass_triggered = False
                continue
            
            return result, True

        # --- 5. Fallback to Original Item ---
        return parser.parse(selected_item["content"]), True

# --- Standard Commands ---

class CmdFile(BaseCommand):
    def __init__(self):
        self.name = "file"
        self.is_container = False

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        args, kwargs = self.parse_args(arg)
        filename = kwargs.get("name", args[0] if args else "")
        rec_val = kwargs.get("recursive", "true").lower()
        recursive = rec_val != "false"
        return parser.load_file_content(filename, recursive=recursive), True

class CmdAll(BaseCommand):
    def __init__(self):
        self.name = "all"
        self.is_container = False

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        args, kwargs = self.parse_args(arg)
        target = kwargs.get("dir", args[0] if args else "")
        rec_val = kwargs.get("recursive", "true").lower()
        recursive = rec_val != "false"
        return parser.load_folder_content(target, recursive=recursive), True

class CmdGet(BaseCommand):
    def __init__(self):
        self.name = "get"
        self.is_container = False

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        args, kwargs = self.parse_args(arg)
        var_name = kwargs.get("var", args[0] if args else "")
        return parser.resolve_var(var_name), False

class CmdSet(BaseCommand):
    def __init__(self):
        self.name = "set"
        self.is_container = True

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        args, kwargs = self.parse_args(arg)
        var_name = kwargs.get("name", args[0] if args else "")
        if var_name:
            val = parser.parse(content)
            parser.vars[var_name] = val
        return "", False

class CmdOverride(BaseCommand):
    def __init__(self):
        self.name = "override"
        self.is_container = True

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        args, kwargs = self.parse_args(arg)
        var_name = kwargs.get("name", args[0] if args else "")
        
        if var_name:
            # Parse the content to allow dynamic values inside the override block
            val = parser.parse(content)
            parser.overrides[var_name] = val
            
        return "", False

class CmdInc(BaseCommand):
    def __init__(self):
        self.name = "inc"
        self.is_container = False
    
    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        args, _ = self.parse_args(arg)
        subject = args[0] if args else ""
        if subject:
            val = parser.resolve_var(subject) or "0"
            try: parser.vars[subject] = str(int(float(val)) + 1)
            except: pass
        return "", False

class CmdDec(BaseCommand):
    def __init__(self):
        self.name = "dec"
        self.is_container = False

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        args, _ = self.parse_args(arg)
        subject = args[0] if args else ""
        if subject:
            val = parser.resolve_var(subject) or "0"
            try: parser.vars[subject] = str(int(float(val)) - 1)
            except: pass
        return "", False

class CmdExists(BaseCommand):
    def __init__(self):
        self.name = "exists"
        self.is_container = False

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        args, _ = self.parse_args(arg)
        subject = args[0] if args else ""
        return "1" if (subject in parser.vars or subject in parser.overrides) else "", False

class CmdLog(BaseCommand):
    def __init__(self):
        self.name = "log"
        self.is_container = False
    
    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        print(f"[Log] Registry Size: {len(parser.registry)}")
        if parser.registry:
            print(f"[Log] Sample Item: {parser.registry[0]}")
        return "", False

class CmdStop(BaseCommand):
    def __init__(self):
        self.name = "stop"
        self.is_container = False
    
    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        parser.stop_triggered = True
        return "", False

class CmdBreak(BaseCommand):
    def __init__(self):
        self.name = "break"
        self.is_container = False
    
    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        parser.break_triggered = True
        return "", False

class CmdCalc(BaseCommand):
    def __init__(self):
        self.name = "calc"
        self.is_container = True

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        expr = parser.parse(content)
        return parser.safe_eval(expr), False

class CmdLen(BaseCommand):
    def __init__(self):
        self.name = "len"
        self.is_container = True

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        target = parser.parse(content)
        return str(len(target)), False

class CmdRange(BaseCommand):
    """
    Generates a random number.
    Usage:
      [range] -> float 0.0 to 1.0 (3 dec)
      [range 10] -> integer 0 to 10
      [range 5 10] -> integer 5 to 10
      [range 0.5 1.5] -> float 0.5 to 1.5 (3 dec)
      [range min=2 max=4] -> integer 2 to 4
    """
    def __init__(self):
        self.name = "range"
        self.is_container = False

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        args, kwargs = self.parse_args(arg)
        
        # Default behavior: float 0.0 to 1.0
        min_v = 0.0
        max_v = 1.0
        is_float_mode = False
        has_args = False

        v1_str = args[0] if len(args) > 0 else kwargs.get("min")
        v2_str = args[1] if len(args) > 1 else kwargs.get("max")

        if v1_str is not None:
            has_args = True
            if '.' in str(v1_str): is_float_mode = True
            
            if v2_str is not None:
                if '.' in str(v2_str): is_float_mode = True
                try:
                    min_v = float(v1_str)
                    max_v = float(v2_str)
                except: pass
            else:
                # One argument usually implies max (0 to Max)
                try:
                    max_v = float(v1_str)
                    min_v = 0.0
                except: pass
        
        if not has_args:
            return f"{parser.rng.uniform(0.0, 1.0):.3f}", False
        
        if is_float_mode:
            return f"{parser.rng.uniform(min_v, max_v):.3f}", False
        else:
            return str(parser.rng.randint(int(min_v), int(max_v))), False

class CmdDef(BaseCommand):
    def __init__(self):
        self.name = "def"
        self.is_container = True

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        args, kwargs = self.parse_args(arg)
        name = kwargs.get("name", args[0] if args else "")
        if name:
            parser.macros[name] = content
        return "", False

class CmdCall(BaseCommand):
    def __init__(self):
        self.name = "call"
        self.is_container = False

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        args, kwargs = self.parse_args(arg)
        name = kwargs.get("name", args[0] if args else "")
        macro = parser.macros.get(name, "")
        return parser.parse(macro), False

class CmdJoin(BaseCommand):
    def __init__(self):
        self.name = "join"
        self.is_container = True

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        args, kwargs = self.parse_args(arg)
        sep = kwargs.get("sep", args[0] if args else "")
        sep = sep.replace('\\n', '\n').replace('\\t', '\t')
        
        processed = parser.parse(content)
        options, _ = parser.parse_list(processed)
        return sep.join(options), False

class CmdShuffle(BaseCommand):
    def __init__(self):
        self.name = "shuffle"
        self.is_container = True

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        args, kwargs = self.parse_args(arg)
        sep = kwargs.get("sep", args[0] if args else "")
        sep = sep.replace('\\n', '\n').replace('\\t', '\t') if sep else ""
        
        processed = parser.parse(content)
        options, auto_sep = parser.parse_list(processed)
        final_sep = sep if sep else auto_sep
        
        parser.rng.shuffle(options)
        return final_sep.join(options), False

class CmdIgnore(BaseCommand):
    def __init__(self):
        self.name = "ignore"
        self.is_container = True

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        return content, False

class CmdMute(BaseCommand):
    def __init__(self):
        self.name = "mute"
        self.is_container = True

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        parser.parse(content)
        if parser.break_triggered: parser.break_triggered = False
        return "", False

class CmdComment(BaseCommand):
    def __init__(self):
        self.name = "comment"
        self.is_container = True
    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        return "", False

class CmdCommentHash(BaseCommand):
    def __init__(self):
        self.name = "#"
        self.is_container = True
    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        return "", False

class CmdLoop(BaseCommand):
    def __init__(self):
        self.name = "loop"
        self.is_container = True

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        args, kwargs = self.parse_args(arg)
        count_str = kwargs.get("count", args[0] if args else "1")
        count = 1
        if count_str.isdigit():
            count = int(count_str)
        
        results = []
        for _ in range(count):
            res = parser.parse(content)
            results.append(res)
            if parser.stop_triggered: break
            if parser.break_triggered:
                parser.break_triggered = False
                break
        return "".join(results), False

class CmdChance(BaseCommand):
    """
    Probabilistic output.
    """
    def __init__(self):
        self.name = "chance"
        self.is_container = True

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        args, kwargs = self.parse_args(arg)
        val = kwargs.get("value", args[0] if args else "")
        threshold = 0.0
        if val:
            try: threshold = float(val.replace('%', '').strip())
            except ValueError: pass
        
        did_pass = (parser.rng.random() * 100) < threshold
        
        output = ""
        if did_pass:
            output = parser.parse(content)
        
        parser.last_condition_result = did_pass
        return output, True

class CmdRan(BaseCommand):
    def __init__(self):
        self.name = "ran"
        self.is_container = True

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        options = parser.split_safe(content, separator='|')
        
        args, kwargs = self.parse_args(arg)
        count = 1
        val = kwargs.get("count", args[0] if args else "")
        if val.isdigit(): count = int(val)
        
        if len(options) > 1:
            if count == 1:
                return parser.parse(parser.rng.choice(options)), True
            else:
                selected = parser.rng.choices(options, k=count) if count > len(options) else parser.rng.sample(options, k=count)
                return ",".join([parser.parse(opt) for opt in selected]), True
        
        lines = parser.split_safe(content, separator='\n')
        # Filter empty lines
        lines = [l for l in lines if l]

        if len(lines) > 1:
            if count == 1:
                return parser.parse(parser.rng.choice(lines)), True
            else:
                selected = parser.rng.choices(lines, k=count) if count > len(lines) else parser.rng.sample(lines, k=count)
                return ",".join([parser.parse(opt) for opt in selected]), True
        
        # Fallback for single item (lines found 1 item, or options found 1 item, which is the whole content)
        if len(lines) == 1:
             return parser.parse(lines[0]), True
        
        return "", False

# --- Weighting Commands ---

def _generate_weight(parser: GrammarParser, args: List[str], kwargs: Dict[str, str]) -> str:
    """Helper to generate a random weight using range logic but with different defaults (1.0-1.4)."""
    # Defaults for weighting
    min_v = 1.0
    max_v = 1.4
    
    # Logic copied from CmdRange (mostly)
    is_float_mode = True # Default to float for weights
    has_args = False

    v1_str = args[0] if len(args) > 0 else kwargs.get("min")
    v2_str = args[1] if len(args) > 1 else kwargs.get("max")

    if v1_str is not None:
        has_args = True
        # If user provides explicit args, we respect them, and switch to int if no dots found?
        # For weighting, usually people want floats. But strict range logic says "int if no dots".
        is_float_mode = False 
        if '.' in str(v1_str): is_float_mode = True
        
        if v2_str is not None:
            if '.' in str(v2_str): is_float_mode = True
            try:
                min_v = float(v1_str)
                max_v = float(v2_str)
            except: pass
        else:
            # For Range, 1 arg = max (0 to Max). 
            # For Weights, if I type [rw 1.5], do I want 0 to 1.5? Probably.
            try:
                max_v = float(v1_str)
                min_v = 0.0
            except: pass
            
    if not has_args:
        # Defaults 1.0 to 1.4 float
        return f"{parser.rng.uniform(min_v, max_v):.3f}"
        
    if is_float_mode:
        return f"{parser.rng.uniform(min_v, max_v):.3f}"
    else:
        return str(parser.rng.randint(int(min_v), int(max_v)))

class CmdIRW(BaseCommand):
    """
    InvokeAI Random Weight.
    Syntax: (content)weight
    Default weight: 1.0 to 1.4
    """
    def __init__(self):
        self.name = "irw"
        self.is_container = True

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        processed_content = parser.parse(content)
        if not processed_content: return "", False
        
        args, kwargs = self.parse_args(arg)
        weight = _generate_weight(parser, args, kwargs)
        
        return f"({processed_content}){weight}", False

class CmdRW(BaseCommand):
    """
    Standard Random Weight.
    Syntax: (content:weight)
    Default weight: 1.0 to 1.4
    """
    def __init__(self):
        self.name = "rw"
        self.is_container = True

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        processed_content = parser.parse(content)
        if not processed_content: return "", False

        args, kwargs = self.parse_args(arg)
        weight = _generate_weight(parser, args, kwargs)
        
        return f"({processed_content}:{weight})", False

# --- Conditional Logic ---

def _eval_condition(parser: GrammarParser, raw_arg: str) -> bool:
    condition_str = raw_arg.strip()
    
    op = None
    parts = []
    
    if '==' in condition_str:
        parts = condition_str.split('==', 1)
        op = '=='
    elif '!=' in condition_str:
        parts = condition_str.split('!=', 1)
        op = '!='
    else:
        key = condition_str
        val = parser.resolve_var(key)
        if val:
            return val != "0" and val.lower() != "false"
        if key == "1": return True
        return False

    var_name = parts[0].strip()
    test_val = parts[1].strip()
    
    if (test_val.startswith('"') and test_val.endswith('"')) or \
       (test_val.startswith("'") and test_val.endswith("'")):
        test_val = test_val[1:-1]
        
    current_val = parser.resolve_var(var_name)
    
    if op == '==': return current_val == test_val
    return current_val != test_val

class CmdIf(BaseCommand):
    def __init__(self):
        self.name = "if"
        self.is_container = True

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        did_pass = _eval_condition(parser, arg)
        
        result = ""
        if did_pass:
            result = parser.parse(content)
            
        parser.last_condition_result = did_pass
        return result, True

class CmdElseIf(BaseCommand):
    def __init__(self):
        self.name = "elseif"
        self.is_container = True

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        if parser.last_condition_result is not False: 
            return "", False
        
        clean_arg = arg.strip()
        args, _ = self.parse_args(arg)
        is_chance = False
        threshold = 0.0
        
        if args and len(args) == 1 and re.match(r'^\d+(\.\d+)?%?$', args[0]):
             try:
                threshold = float(args[0].replace('%', ''))
                is_chance = True
             except: pass
        
        did_pass = False
        if is_chance:
            if (parser.rng.random() * 100) < threshold:
                did_pass = True
        else:
            if _eval_condition(parser, clean_arg):
                did_pass = True
        
        result = ""
        if did_pass:
            result = parser.parse(content)
            
        parser.last_condition_result = did_pass
        return result, True

class CmdElse(BaseCommand):
    def __init__(self):
        self.name = "else"
        self.is_container = True

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        if parser.last_condition_result is False:
            parser.last_condition_result = True
            return parser.parse(content), True
        return "", False

class CmdSwitch(BaseCommand):
    def __init__(self):
        self.name = "switch"
        self.is_container = True

    def execute(self, parser: GrammarParser, arg: str, content: str) -> Tuple[str, bool]:
        args, kwargs = self.parse_args(arg)
        var_name = kwargs.get("var", args[0] if args else "")
        var_val = parser.resolve_var(var_name)
        
        cursor = 0
        matched_content = None
        default_content = ""

        while cursor < len(content):
            match = re.search(r'\[(case|default)(?:\s+([^\]]+))?\]', content[cursor:])
            if not match: break
            
            tag_type = match.group(1)
            case_val = (match.group(2) or "").strip()
            if (case_val.startswith('"') and case_val.endswith('"')): case_val = case_val[1:-1]
            
            tag_start = cursor + match.start()
            tag_end_head = cursor + match.end()
            close_tag = f"[/{tag_type}]"
            
            block_end = parser._find_closing_tag(content, tag_end_head, f"[{tag_type}", close_tag)
            if block_end == -1:
                cursor = tag_end_head
                continue

            inner = content[tag_end_head:block_end]
            
            if tag_type == 'case':
                if var_val == case_val:
                    matched_content = inner
                    break
            elif tag_type == 'default':
                default_content = inner

            cursor = block_end + len(close_tag)

        result = matched_content if matched_content is not None else default_content
        return parser.parse(result), True