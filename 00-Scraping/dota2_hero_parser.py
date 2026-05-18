import json
import re
import math
import requests
from typing import Dict, List, Any, Optional

# WIKI API CONFIGURATION
WIKI_BASE_URL = "https://dota2.fandom.com"
WIKI_API_URL = f"{WIKI_BASE_URL}/api.php"
WIKI_HEADERS = {'User-Agent': 'Dota2-Hero-Parser/1.0'}


def fetch_raw_wikitext(page_title: str) -> Optional[str]:
    """
    Fetches raw wiki text from the Dota 2 Wiki API.
    """
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": page_title,
        "rvprop": "content",
        "rvslots": "main",
        "format": "json"
    }
    try:
        response = requests.get(WIKI_API_URL, params=params, headers=WIKI_HEADERS)
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1":
                return None
            return page_data["revisions"][0]["slots"]["main"]["*"]
    except Exception as e:
        print(f"Error fetching {page_title}: {e}")
        return None


class MediaWikiTemplateEvaluator:
    """
    Advanced MediaWiki evaluator for Dota 2 Wiki patterns.
    """
    
    CONSTANTS = {
        "bonus universal damage": 0.7,
        "bonus strength damage": 1.0,
        "bonus agility damage": 1.0,
        "bonus intelligence damage": 1.0,
        "bonus health": 22.0,
        "bonus mana": 12.0,
        "bonus health regeneration flat": 0.09,
        "bonus mana regeneration flat": 0.05,
        "bonus armor": (1/6.0),
        "bonus magic resistance": 0.001,
        "tick rate": 1,
        "magic resistance": 0.25,
        "base magic resistance": 0.25,
    }
    
    def __init__(self):
        self.variables = {}
        self.talent_values = {}
        self.internal_cache = {}
        self.hero_name = None
        
    def set_hero_name(self, name: str):
        self.hero_name = name

    def extract_variables_from_wiki_text(self, wiki_text: str):
        """
        Extract variables from wiki text in two passes:
        1. First pass: Extract simple values (numbers, percentages)
        2. Second pass: Evaluate complex template values
        """
        current_ability = None
        ability_name_pattern = r'\|\s*name\s*=\s*(.+)'
        value_pattern = r'\|\s*(value\d+)(?:\s+(\w+))?\s*=\s*(.+)'
        cooldown_pattern = r'\|\s*cooldown\s*=\s*(.+)'
        mana_pattern = r'\|\s*mana\s*=\s*(.+)'
        
        # Store complex values for second pass
        complex_values = {}
        
        lines = wiki_text.split('\n')
        for line in lines:
            name_match = re.search(ability_name_pattern, line)
            if name_match:
                current_ability = name_match.group(1).strip()
                continue
            
            # Extract cooldown
            cooldown_match = re.search(cooldown_pattern, line)
            if cooldown_match and current_ability:
                cooldown_val = cooldown_match.group(1).strip()
                cooldown_val = re.sub(r'\s*\{\{.*$', '', cooldown_val)
                self.variables[f"{current_ability} cooldown"] = self._extract_base_value(cooldown_val)
            
            # Extract mana
            mana_match = re.search(mana_pattern, line)
            if mana_match and current_ability:
                mana_val = mana_match.group(1).strip()
                mana_val = re.sub(r'\s*\{\{.*$', '', mana_val)
                self.variables[f"{current_ability} mana"] = self._extract_base_value(mana_val)
            
            value_match = re.search(value_pattern, line)
            if value_match and current_ability:
                v_key = value_match.group(1)
                modifier = value_match.group(2) or ""
                content = value_match.group(3).strip()
                
                var_name = f"{current_ability} {v_key}"
                if modifier:
                    var_name += f" {modifier}"
                
                # Check for STANDALONE vardefineecho (the entire content is just vardefineecho)
                vde_match = re.match(r'^\{\{#vardefineecho:([^|]+)\|([^}]+)\}\}$', content)
                if vde_match:
                    defined_var = vde_match.group(1).strip()
                    defined_value = vde_match.group(2).strip()
                    self.variables[defined_var] = defined_value
                    self.variables[var_name] = defined_value
                # Check if it contains templates that need evaluation
                elif '{{' in content:
                    # Store for second pass
                    complex_values[var_name] = content
                else:
                    self.variables[var_name] = self._extract_base_value(content)
        
        # Second pass: Evaluate complex template values
        for var_name, content in complex_values.items():
            evaluated = self.evaluate(content)
            # Handle percentage result
            if evaluated.endswith('%'):
                try:
                    num = float(evaluated.rstrip('%'))
                    evaluated = str(num / 100)
                except ValueError:
                    pass
            self.variables[var_name] = evaluated
        
        self._extract_talent_values(wiki_text)

    def _extract_base_value(self, value: str) -> str:
        """Extract the base value. Keep percentages as-is for template compatibility."""
        value = value.strip()
        
        # If it's a simple value with optional percentage
        if re.match(r'^[\d.]+%?$', value):
            return value
        
        # If it's a slash-separated value
        if re.match(r'^[\d.]+%?(/[\d.]+%?)+$', value):
            return value
        
        # If it's a simple number or slash-separated numbers
        if re.match(r'^[\d./% -]+$', value):
            return value
        
        # Try to find a leading number/percentage pattern
        match = re.match(r'^([\d.]+%?)', value)
        if match:
            return match.group(1)
        
        # Return as-is
        return value

    def _extract_talent_values(self, wiki_text: str):
        talent_refs = re.findall(r'\{\{Show\|T\|(\w+)\|([^}|]+)(?:\|[^}]+)?\}\}', wiki_text)
        for hero, talent_ref in talent_refs:
            if self.hero_name and (hero.lower() == self.hero_name.lower()):
                self.talent_values[talent_ref.strip()] = f"[TALENT:{talent_ref.strip()}]"

    def evaluate(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        
        result = text
        result = self._remove_symbol_parentheticals(result)
        
        max_iterations = 30
        for _ in range(max_iterations):
            old_result = result
            
            result = self._evaluate_g(result)
            result = self._evaluate_cf(result)
            result = self._evaluate_cl(result)
            result = self._evaluate_u(result)
            result = self._evaluate_keypress(result)
            result = self._evaluate_attribute_id(result)
            result = self._evaluate_loop(result)
            result = self._evaluate_vardefine(result)
            result = self._evaluate_vardefineecho(result)
            result = self._evaluate_var(result)
            result = self._evaluate_explode(result)
            result = self._evaluate_calc(result)
            result = self._evaluate_expr(result)
            result = self._evaluate_show(result)
            result = self._evaluate_value_color(result)
            
            if result == old_result:
                break
        
        return self._clean_placeholders(result)

    def _remove_symbol_parentheticals(self, text: str) -> str:
        symbols_to_remove = [
            '{{Symbol|Talent}}', 
            '{{Symbol|Aghshard}}', 
            '{{Symbol|Aghanim}}',
            '{{Symbol|Daytime}}',
            '{{Symbol|Nighttime}}',
        ]
        
        result = []
        i = 0
        
        while i < len(text):
            if text[i] == '(' and i + 1 < len(text):
                remaining = text[i+1:].lstrip()
                should_remove = any(remaining.startswith(sym) for sym in symbols_to_remove)
                
                if should_remove:
                    paren_count = 1
                    j = i + 1
                    
                    while j < len(text) and paren_count > 0:
                        if text[j] == '(':
                            paren_count += 1
                        elif text[j] == ')':
                            paren_count -= 1
                        j += 1
                    
                    i = j
                    continue
            
            result.append(text[i])
            i += 1
        
        # Also remove standalone Symbol templates (not in parentheses)
        result_text = ''.join(result)
        result_text = re.sub(r'\{\{Symbol\|(?:Daytime|Nighttime)\}\}\s*', '', result_text)
        
        return result_text

    def _evaluate_loop(self, text: str) -> str:
        pattern_start = '{{#loop:'
        
        while pattern_start in text:
            start_idx = text.find(pattern_start)
            if start_idx == -1:
                break
            
            content_start = start_idx + len(pattern_start)
            brace_count = 2
            j = content_start
            
            while j < len(text) and brace_count > 0:
                if text[j:j+2] == '{{':
                    brace_count += 2
                    j += 2
                elif text[j:j+2] == '}}':
                    brace_count -= 2
                    j += 2
                else:
                    j += 1
            
            if brace_count != 0:
                break
            
            loop_content = text[content_start:j-2]
            parts = self._split_loop_params(loop_content)
            
            if len(parts) >= 4:
                var_name = parts[0].strip()
                try:
                    start_val = int(parts[1].strip())
                    count = int(parts[2].strip())
                except ValueError:
                    text = text[:start_idx] + text[j:]
                    continue
                
                template = '|'.join(parts[3:])
                
                outputs = []
                for iteration in range(count):
                    current_val = start_val + iteration
                    
                    old_val = self.internal_cache.get(var_name)
                    self.internal_cache[var_name] = str(current_val)
                    self.variables[var_name] = str(current_val)
                    
                    evaluated = self.evaluate(template)
                    outputs.append(evaluated)
                    
                    if old_val is not None:
                        self.internal_cache[var_name] = old_val
                        self.variables[var_name] = old_val
                    else:
                        self.internal_cache.pop(var_name, None)
                        self.variables.pop(var_name, None)
                
                result = ' '.join(outputs)
                text = text[:start_idx] + result + text[j:]
            else:
                text = text[:start_idx] + text[j:]
        
        return text

    def _split_loop_params(self, content: str) -> List[str]:
        parts = []
        current = []
        brace_count = 0
        
        for char in content:
            if char == '{':
                brace_count += 1
                current.append(char)
            elif char == '}':
                brace_count -= 1
                current.append(char)
            elif char == '|' and brace_count == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(char)
        
        if current:
            parts.append(''.join(current))
        
        return parts

    def _evaluate_vardefine(self, text: str) -> str:
        result = text
        pattern_start = '{{#vardefine:'
        
        while pattern_start in result:
            start_idx = result.find(pattern_start)
            if start_idx == -1:
                break
            
            name_start = start_idx + len(pattern_start)
            pipe_idx = result.find('|', name_start)
            if pipe_idx == -1:
                break
            
            var_name = result[name_start:pipe_idx].strip()
            
            value_start = pipe_idx + 1
            brace_count = 2
            j = value_start
            
            while j < len(result) and brace_count > 0:
                if result[j:j+2] == '{{':
                    brace_count += 2
                    j += 2
                elif result[j:j+2] == '}}':
                    brace_count -= 2
                    j += 2
                else:
                    j += 1
            
            if brace_count == 0:
                value_content = result[value_start:j-2].strip()
                evaluated_value = self.evaluate(value_content)
                self.internal_cache[var_name] = evaluated_value
                self.variables[var_name] = evaluated_value
                result = result[:start_idx] + result[j:]
            else:
                break
        
        return result

    def _evaluate_u(self, text: str) -> str:
        return re.sub(r'\{\{U\|[^|]+\|text=([^}|]+)\}\}', r'\1', text)

    def _evaluate_keypress(self, text: str) -> str:
        text = text.replace('{{Key press|double}}', 'double-tapping')
        return re.sub(r'\{\{Key press\|([^}]+)\}\}', r'\1', text)

    def _evaluate_calc(self, text: str) -> str:
        # Pattern: {{calc|expr|params}} or {{calc|expr|params|%}}
        pattern = r'\{\{calc\|([^|]+)\|([^}]+)\}\}'
        
        def replace(match):
            expr_part = match.group(1).strip()
            params_raw = match.group(2).strip()
            
            # Check for percentage flag at the end
            as_percentage = False
            if params_raw.endswith('|%'):
                as_percentage = True
                params_raw = params_raw[:-2].strip()
            elif '|%' in params_raw:
                # Handle |%}} at end
                params_raw = params_raw.replace('|%', '')
                as_percentage = True
            
            rnd = None
            round_match = re.match(r'(.+?)\s+round\s*(\d+)\s*$', expr_part)
            if round_match:
                expr_part = round_match.group(1).strip()
                rnd = int(round_match.group(2))
            
            params = {}
            parts = self._split_params(params_raw)
            for part in parts:
                if '=' in part:
                    k, v = part.split('=', 1)
                    params[k.strip()] = v.strip()
                elif part.strip() == '%':
                    as_percentage = True
            
            result = self._evaluate_calc_expression(expr_part, params, rnd, as_percentage)
            return result
        
        return re.sub(pattern, replace, text)

    def _split_params(self, params_str: str) -> List[str]:
        parts = []
        current = []
        brace_count = 0
        
        for char in params_str:
            if char == '{':
                brace_count += 1
                current.append(char)
            elif char == '}':
                brace_count -= 1
                current.append(char)
            elif char == '|' and brace_count == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(char)
        
        if current:
            parts.append(''.join(current))
        
        return parts

    def _evaluate_calc_expression(self, expression: str, params: Dict[str, str], rnd: int = None, as_percentage: bool = False) -> str:
        level_count = 1
        for v in params.values():
            if '/' in str(v):
                level_count = max(level_count, len(str(v).split('/')))
        
        safe_dict = {
            "ceil": math.ceil,
            "floor": math.floor,
            "round": round,
            "trunc": math.trunc,
            "abs": abs,
            "pow": pow,
            "min": min,
            "max": max,
        }

        results = []
        for level in range(level_count):
            current_vars = {}
            
            for k, v in params.items():
                str_v = str(v)
                if '/' in str_v:
                    parts = str_v.split('/')
                    val = parts[min(level, len(parts)-1)]
                else:
                    val = str_v
                
                # Handle percentage values - convert to decimal
                if val.endswith('%'):
                    try:
                        current_vars[k] = float(val.rstrip('%')) / 100
                        continue
                    except ValueError:
                        pass
                
                try:
                    current_vars[k] = float(val)
                except ValueError:
                    current_vars[k] = 0.0

            # Also convert any percentages in the expression itself
            expr_to_eval = expression
            def convert_pct(m):
                try:
                    return str(float(m.group(1)) / 100)
                except ValueError:
                    return m.group(0)
            expr_to_eval = re.sub(r'(\d+(?:\.\d+)?)%', convert_pct, expr_to_eval)

            try:
                res = eval(expr_to_eval, {"__builtins__": {}}, {**safe_dict, **current_vars})
                
                if isinstance(res, (int, float)):
                    # Convert to percentage if requested
                    if as_percentage:
                        res = res * 100
                    
                    if rnd is not None:
                        res = round(res, rnd)
                    
                    if isinstance(res, float) and res == int(res):
                        formatted = str(int(res))
                    elif rnd is not None:
                        fmt = f"{{:.{rnd}f}}"
                        formatted = fmt.format(res)
                    else:
                        formatted = f"{res:.4f}".rstrip('0').rstrip('.')
                    
                    # Add % suffix if percentage
                    if as_percentage:
                        formatted += '%'
                    
                    results.append(formatted)
                else:
                    results.append(str(res))
            except Exception:
                results.append("?")
        
        return '/'.join(results)

    def _evaluate_var(self, text: str) -> str:
        pattern = r'\{\{#var:([^}]+)\}\}'
        
        def replace(match):
            name = match.group(1).strip()
            val = self.internal_cache.get(name) or self.variables.get(name)
            return str(val) if val is not None else f"[VAR:{name}]"
        
        return re.sub(pattern, replace, text)

    def _evaluate_g(self, text: str) -> str:
        pattern = r'\{\{G\|([^}]+)\}\}'
        
        def replace(match):
            const_name = match.group(1).strip().lower()
            val = self.CONSTANTS.get(const_name)
            return str(val) if val is not None else f"[G:{match.group(1)}]"
        
        return re.sub(pattern, replace, text)

    def _evaluate_explode(self, text: str) -> str:
        """
        Evaluate {{#explode:value|delimiter}} or {{#explode:value|delimiter|index}}.
        
        When no index specified, defaults to index 0 (first part).
        """
        pattern = r'\{\{#explode:([^|]+)\|([^|}]+)(?:\|(\d+))?\}\}'
        
        def replace(match):
            value = match.group(1).strip()
            delimiter = match.group(2).strip()
            index_str = match.group(3)
            
            # Default to index 0 if not specified
            idx = int(index_str) if index_str is not None else 0
            
            if delimiter in value:
                parts = value.split(delimiter)
                if idx < len(parts):
                    return parts[idx]
            
            return value
        
        return re.sub(pattern, replace, text)

    def _evaluate_cf(self, text: str) -> str:
        text = re.sub(r'\{\{cf\|([^|}]+)\|([^}]+)\}\}', r'(\1) \2', text)
        text = re.sub(r'\{\{cf\|([^|}]+)\}\}', r'(\1)', text)
        return text

    def _evaluate_cl(self, text: str) -> str:
        """
        Evaluate {{Cl|type|name|hero}} -> name (type)
        """
        pattern = r'\{\{Cl\|([^|}]+)\|([^|}]+)(?:\|[^}]*)?\}\}'
        
        def replace(match):
            item_type = match.group(1).strip()
            item_name = match.group(2).strip()
            return f"{item_name} ({item_type})"
        
        return re.sub(pattern, replace, text)

    def _evaluate_attribute_id(self, text: str) -> str:
        """
        Evaluate {{Attribute ID|strength}} -> strength
        """
        pattern = r'\{\{Attribute ID\|([^}]+)\}\}'
        return re.sub(pattern, r'\1', text)

    def _evaluate_vardefineecho(self, text: str) -> str:
        """
        Evaluate {{#vardefineecho:name|value}} - defines a variable AND returns the value.
        Handles nested templates in the value.
        """
        result = text
        pattern_start = '{{#vardefineecho:'
        
        while pattern_start in result:
            start_idx = result.find(pattern_start)
            if start_idx == -1:
                break
            
            name_start = start_idx + len(pattern_start)
            pipe_idx = result.find('|', name_start)
            if pipe_idx == -1:
                break
            
            var_name = result[name_start:pipe_idx].strip()
            
            value_start = pipe_idx + 1
            brace_count = 2  # We've seen {{#vardefineecho
            j = value_start
            
            while j < len(result) and brace_count > 0:
                if result[j:j+2] == '{{':
                    brace_count += 2
                    j += 2
                elif result[j:j+2] == '}}':
                    brace_count -= 2
                    j += 2
                else:
                    j += 1
            
            if brace_count == 0:
                value_content = result[value_start:j-2].strip()
                # Evaluate the value content (it may contain nested templates)
                evaluated_value = self.evaluate(value_content)
                self.internal_cache[var_name] = evaluated_value
                self.variables[var_name] = evaluated_value
                # Replace with the evaluated value (echo it)
                result = result[:start_idx] + evaluated_value + result[j:]
            else:
                break
        
        return result

    def _evaluate_expr(self, text: str) -> str:
        pattern = r'\{\{#expr:([^}]+)\}\}'
        
        def replace(match):
            expr = match.group(1).strip()
            
            # Handle trunc
            expr = re.sub(r'\btrunc\s*\(', 'int(', expr)
            
            # Handle degree symbol - remove it (trig functions expect radians)
            # The wiki format is: value|° where ° is the delimiter for #explode
            # After #explode, we just have the number
            expr = expr.replace('°', '')
            
            # Handle "round N" at end of expression
            rnd = None
            round_match = re.search(r'\s+round\s*(\d+)\s*$', expr)
            if round_match:
                rnd = int(round_match.group(1))
                expr = expr[:round_match.start()]
            
            safe_dict = {
                "ceil": math.ceil,
                "floor": math.floor,
                "round": round,
                "int": int,
                "abs": abs,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "asin": math.asin,
                "acos": math.acos,
                "atan": math.atan,
                "sqrt": math.sqrt,
                "pi": math.pi,
                "e": math.e,
            }
            
            try:
                result = eval(expr, {"__builtins__": {}}, safe_dict)
                if rnd is not None:
                    result = round(result, rnd)
                if isinstance(result, float) and result == int(result):
                    return str(int(result))
                elif rnd is not None:
                    return f"{result:.{rnd}f}"
                return str(result)
            except Exception as e:
                return f"[EXPR:{expr}]"
        
        return re.sub(pattern, replace, text)

    def _evaluate_show(self, text: str) -> str:
        ability_pattern = r'\{\{Show\|A\|[^|]+\|([^|]+)\|([^}|]+)(?:\|[^}]+)?\}\}'
        
        def replace_ability(match):
            ability = match.group(1).strip()
            value_key = match.group(2).strip()
            var_name = f"{ability} {value_key}"
            return str(self.variables.get(var_name, f"[{ability}:{value_key}]"))
        
        text = re.sub(ability_pattern, replace_ability, text)
        
        talent_pattern = r'\{\{Show\|T\|[^|]+\|([^}|]+)(?:\s+value)?\}\}'
        
        def replace_talent(match):
            talent_ref = match.group(1).strip()
            return str(self.talent_values.get(talent_ref, f"[TALENT:{talent_ref}]"))
        
        return re.sub(talent_pattern, replace_talent, text)

    def _evaluate_value_color(self, text: str) -> str:
        text = re.sub(r'\{\{ValueColor\|\d+\|([^|}]*)\|[^}]+\}\}', r'\1', text)
        text = re.sub(r'\{\{ValueColor\|\d+\|([^|}]+)\}\}', r'\1', text)
        text = re.sub(r'\{\{ValueColor\|[^}]*\}\}', '', text)
        return text

    def _clean_placeholders(self, text: str) -> str:
        max_iterations = 10
        for _ in range(max_iterations):
            old_text = text
            text = re.sub(r'\[(?:CALC|VAR|EXPR|TALENT|G):[^\[\]]*\]', '', text)
            if text == old_text:
                break
        return text


class EnhancedHeroWikiParser:
    """Enhanced wiki parser that properly evaluates MediaWiki templates."""
    
    def __init__(self, raw_wiki_text: str, excluded_sections: Optional[List[str]] = None):
        self.raw_wiki_text = raw_wiki_text
        self.excluded_sections = excluded_sections or [
            "Dota Plus Progress", "Gallery", "Equipment", "References",
            "Recent Changes", "Recommended Items", "Talents"
        ]
        self.sections = {}
        self.parsed_data = {}
        self.evaluator = MediaWikiTemplateEvaluator()
        
        self._pre_extract_hero_name()
        self.evaluator.extract_variables_from_wiki_text(raw_wiki_text)
    
    def _pre_extract_hero_name(self):
        match = re.search(r'\|\s*title\s*=\s*(.+)', self.raw_wiki_text)
        if match:
            hero_name = match.group(1).strip()
            self.evaluator.set_hero_name(hero_name)
    
    def parse(self) -> Dict[str, Any]:
        self._split_into_sections()
        self._parse_hero_infobox()
        self._parse_bio()
        self._parse_abilities()
        self._parse_trivia()
        return self.parsed_data
    
    def _split_into_sections(self):
        section_pattern = r'^==\s*(.+?)\s*==\s*$'
        lines = self.raw_wiki_text.split('\n')
        current_section = "header"
        current_content = []
        
        for line in lines:
            section_match = re.match(section_pattern, line)
            if section_match:
                if current_content:
                    self.sections[current_section] = '\n'.join(current_content)
                section_name = section_match.group(1).strip()
                if section_name not in self.excluded_sections:
                    current_section = section_name
                    current_content = []
                else:
                    current_section = None
                    current_content = []
            else:
                if current_section is not None:
                    current_content.append(line)
        
        if current_section and current_content:
            self.sections[current_section] = '\n'.join(current_content)
    
    def _extract_infobox_value(self, pattern: str, text: str, default: Any = None) -> Any:
        match = re.search(pattern, text, re.MULTILINE)
        return match.group(1).strip() if match else default
    
    def _clean_wiki_markup(self, text: str) -> str:
        if not text:
            return text
        
        text = self.evaluator.evaluate(text)
        
        cleanup_patterns = [
            (r'<sm2>.*?</sm2>', ''),
            (r'\{\{Key press\|([^}]+)\}\}', r'\1'),
            (r'\{\{Key special\|([^}]+)\}\}', r'\1'),
            (r'\[\[([^\]|]+)\]\]', r'\1'),
            (r'\[\[.*?\|([^\]]+)\]\]', r'\1'),
            (r'\{\{([A-Z][a-z]+)\}\}', r'\1'),
            (r'\{\{I\|([^}]+)\}\}', r'\1'),
            (r'\{\{[HUI]\|([^}]+)\}\}', r'\1'),
            (r'\{\{A\|([^|]+)\|[^}]+\}\}', r'\1'),
            (r'\{\{A\|[^|]+\|[^|]+\|text=([^}]+)\}\}', r'\1'),
            (r'\{\{cf\|([^}]+)\}\}', r'(\1)'),
            (r'\{\{tooltip\|([^|]+)\|[^}]+\}\}', r'\1'),
            (r'\{\{Symbol\|[^}]*\}\}', ''),
            (r'\{\{ValueColor\|\d+\|([^|}]+)(?:\|[^}]*)?\}\}', r'\1'),
            (r'\{\{ValueColor\|[^}]*\}\}', ''),
            (r'<code>(.*?)</code>', r'\1'),
            (r'<br\s*/?>', ' '),
            (r'<ref>.*?</ref>', ''),
            (r"'''([^']+)'''", r'\1'),
            (r"''([^']+)''", r'\1'),
            (r'\|[\w]+\}\}', ''),
            (r'\}\}', ''),
            (r'\s+', ' '),
            (r'\(\s*\)', ''),
            (r'\[\s*\]', ''),
            (r'\s*\*\*\*\s*', ' '),
        ]
        
        for pattern, replacement in cleanup_patterns:
            text = re.sub(pattern, replacement, text)
        
        return text.strip()

    def _calculate_derived_stats(self):
        attributes = self.parsed_data.get("attributes", {})
        combat_stats = self.parsed_data.get("combat_stats", {})
        
        if not attributes:
            return
        
        strength = attributes.get("strength", 0)
        agility = attributes.get("agility", 0)
        intelligence = attributes.get("intelligence", 0)
        primary_attr = attributes.get("primary_attribute", "")
        
        attributes["health"] = 120 + strength * 22
        attributes["mana"] = 75 + intelligence * 12
        
        attack_damage_min = combat_stats.get("attack_damage_min", 0)
        attack_damage_max = combat_stats.get("attack_damage_max", 0)
        base_health_regen = combat_stats.get("health_regen", 0)
        base_mana_regen = combat_stats.get("mana_regen", 0)
        base_armor = combat_stats.get("armor", 0)
        
        if primary_attr == "Universal":
            attributes["damage_min"] = round(0.7 * (strength + agility + intelligence) + attack_damage_min, 1)
            attributes["damage_max"] = round(0.7 * (strength + agility + intelligence) + attack_damage_max, 1)
        elif primary_attr == "Strength":
            attributes["damage_min"] = strength + attack_damage_min
            attributes["damage_max"] = strength + attack_damage_max
        elif primary_attr == "Agility":
            attributes["damage_min"] = agility + attack_damage_min
            attributes["damage_max"] = agility + attack_damage_max
        elif primary_attr == "Intelligence":
            attributes["damage_min"] = intelligence + attack_damage_min
            attributes["damage_max"] = intelligence + attack_damage_max
        
        attributes["health_regen"] = round(0.09 * strength + base_health_regen, 2)
        attributes["mana_regen"] = round(0.05 * intelligence + base_mana_regen, 2)
        attributes["armor"] = round(base_armor + (agility / 6), 2)
        
        self.parsed_data["attributes"] = attributes
    
    def _parse_hero_infobox(self):
        header_section = self.sections.get("header", "")
        
        hero_name = self._extract_infobox_value(r'\|\s*title\s*=\s*(.+)', header_section)
        
        hero_data = {
            "name": hero_name,
            "internal_name": self._extract_infobox_value(r'\|\s*intern\s*=\s*(.+)', header_section),
            "hero_id": None,
        }
        
        hero_id = self._extract_infobox_value(r'\|\s*hid\s*=\s*(\d+)', header_section)
        if hero_id:
            hero_data["hero_id"] = int(hero_id)
        
        release_beta = self._extract_infobox_value(r'\|\s*releasedate\s*=\s*(.+?);', header_section)
        release_allstars = self._extract_infobox_value(r'\|\s*allstars\s*=\s*(.+)', header_section)
        if release_beta or release_allstars:
            hero_data["release_date"] = {}
            if release_beta:
                hero_data["release_date"]["beta"] = release_beta
            if release_allstars:
                hero_data["release_date"]["allstars"] = release_allstars
        
        attributes = {}
        primary_attr = self._extract_infobox_value(r'\|\s*primary attribute\s*=\s*(.+)', header_section)
        if primary_attr:
            attributes["primary_attribute"] = primary_attr
        
        for attr in ["strength", "agility", "intelligence"]:
            val = self._extract_infobox_value(rf'\|\s*{attr}\s*=\s*(\d+)', header_section)
            if val:
                attributes[attr] = int(val)
            growth = self._extract_infobox_value(rf'\|\s*{attr} growth\s*=\s*([\d.]+)', header_section)
            if growth:
                attributes[f"{attr}_growth"] = float(growth)
        
        if attributes:
            self.parsed_data["attributes"] = attributes
        
        combat_stats = {}
        stat_patterns = {
            "attack_damage_min": (r'\|\s*attack damage min\s*=\s*(\d+)', int),
            "attack_damage_max": (r'\|\s*attack damage max\s*=\s*(\d+)', int),
            "health_regen": (r'\|\s*health regen\s*=\s*([\d.]+)', float),
            "mana_regen": (r'\|\s*mana regen\s*=\s*([\d.]+)', float),
            "armor": (r'\|\s*armor\s*=\s*(-?[\d.]+)', float),
            "movement_speed": (r'\|\s*movement speed\s*=\s*(\d+)', int),
            "attack_speed": (r'\|\s*attack speed\s*=\s*(\d+)', int),
            "attack_range": (r'\|\s*attack range\s*=\s*(\d+)', int),
            "attack_point": (r'\|\s*attack point\s*=\s*([\d.]+)', float),
            "attack_backswing": (r'\|\s*attack backswing\s*=\s*([\d.]+)', float),
            "base_attack_time": (r'\|\s*base attack time\s*=\s*([\d.]+)', float),
            "projectile_speed": (r'\|\s*projectile speed\s*=\s*(\d+)', int),
            "sight_range_day": (r'\|\s*sight range day\s*=\s*(\d+)', int),
            "sight_range_night": (r'\|\s*sight range night\s*=\s*(\d+)', int),
            "turn_rate": (r'\|\s*turn rate\s*=\s*([\d.]+)', float),
            "collision_size": (r'\|\s*collision size\s*=\s*(\d+)', int),
            "bound_radius": (r'\|\s*bound radius\s*=\s*(\d+)', int),
        }
        
        for stat_name, (pattern, converter) in stat_patterns.items():
            val = self._extract_infobox_value(pattern, header_section)
            if val:
                combat_stats[stat_name] = converter(val)
        
        range_type = self._extract_infobox_value(r'\|\s*range type\s*=\s*(.+)', header_section)
        if range_type:
            combat_stats["range_type"] = range_type
        
        gib_type = self._extract_infobox_value(r'\|\s*gib type\s*=\s*(.+)', header_section)
        if gib_type:
            combat_stats["gib_type"] = gib_type
        
        if combat_stats:
            self.parsed_data["combat_stats"] = combat_stats
        
        self._calculate_derived_stats()
        
        gameplay = {}
        gameplay_intro = self._extract_infobox_value(r'\|\s*intro\s*=\s*(.+)', header_section)
        if gameplay_intro:
            gameplay["intro"] = gameplay_intro
        
        complexity = self._extract_infobox_value(r'\|\s*complexity\s*=\s*(\d+)', header_section)
        if complexity:
            gameplay["complexity"] = int(complexity)
        
        roles_text = self._extract_infobox_value(r'\|\s*roles\s*=\s*(.+)', header_section)
        if roles_text:
            gameplay["roles"] = [r.strip() for r in roles_text.split(',')]
        
        adjectives_text = self._extract_infobox_value(r'\|\s*adjectives\s*=\s*(.+)', header_section)
        if adjectives_text:
            gameplay["adjectives"] = [a.strip() for a in adjectives_text.split(',')]
        
        legs = self._extract_infobox_value(r'\|\s*legs\s*=\s*(\d+)', header_section)
        if legs:
            gameplay["legs"] = int(legs)
        
        if gameplay:
            self.parsed_data["gameplay"] = gameplay
        
        self.parsed_data["hero"] = hero_data
    
    def _parse_bio(self):
        bio_section = self.sections.get("Bio", "")
        if not bio_section:
            return
        
        for field, key in [("name", "real_name"), ("alias", "alias"), 
                           ("title", "title"), ("quote", "quote"), ("voice", "voice_actor")]:
            val = self._extract_infobox_value(rf'\|\s*{field}\s*=\s*(.+)', bio_section)
            if val:
                self.parsed_data["hero"][key] = val
        
        lore_match = re.search(r'\|\s*lore\s*=\s*(.+?)(?=\n\}\})', bio_section, re.DOTALL)
        if lore_match:
            lore = lore_match.group(1).strip()
            lore = re.sub(r'<br><br>', '\n\n', lore)
            self.parsed_data["hero"]["lore"] = lore
    
    def _extract_ability_blocks(self, section_content: str) -> List[str]:
        blocks = []
        i = 0
        
        while i < len(section_content):
            start_match = re.search(r'\{\{Ability\s*\n', section_content[i:])
            if not start_match:
                break
            
            start_pos = i + start_match.end()
            brace_count = 2
            j = start_pos
            
            while j < len(section_content) and brace_count > 0:
                if section_content[j:j+2] == '{{':
                    brace_count += 2
                    j += 2
                elif section_content[j:j+2] == '}}':
                    brace_count -= 2
                    j += 2
                else:
                    j += 1
            
            if brace_count == 0:
                blocks.append(section_content[start_pos:j-2])
            
            i = j
        
        return blocks
    
    def _parse_ability_block(self, ability_text: str) -> Dict[str, Any]:
        ability = {}
        
        for field, key in [("ID", "id"), ("intern", "internal_name"), ("name", "name"), 
                           ("type", "type"), ("key", "key")]:
            val = self._extract_infobox_value(rf'\|\s*{field}\s*=\s*(.+)', ability_text)
            if val:
                ability[key] = val
        
        ability_name = ability.get("name", "")
        
        desc_match = re.search(r'\|\s*description\s*=\s*(.+?)(?=\n\|)', ability_text, re.DOTALL)
        if desc_match:
            desc = desc_match.group(1).strip()
            desc = self.evaluator.evaluate(desc)
            desc = self._clean_wiki_markup(desc)
            ability["description"] = desc
        
        for field, key in [("target", "target"), ("target2", "target2")]:
            val = self._extract_infobox_value(rf'\|\s*{field}\s*=\s*(.+)', ability_text)
            if val:
                ability[key] = val
        
        for field, key in [("affects", "affects"), ("affects2", "affects2")]:
            val = self._extract_infobox_value(rf'\|\s*{field}\s*=\s*(.+)', ability_text)
            if val:
                ability[key] = [a.strip() for a in val.split(',')] if ',' in val else val
        
        damage_type = self._extract_infobox_value(r'\|\s*damagetype\s*=\s*(.+)', ability_text)
        if damage_type:
            ability["damage_type"] = damage_type
        
        bool_fields = {
            "piercesdbi": "pierces_debuff_immunity",
            "linkenblock": "linken_sphere_block",
            "breakable": "breakable",
            "illusionuse": "illusion_use",
            "oncastproc": "on_cast_proc"
        }
        
        for field, key in bool_fields.items():
            val = self._extract_infobox_value(rf'\|\s*{field}\s*=\s*(.+)', ability_text)
            if val:
                ability[key] = val.lower() != "no" if field in ["breakable", "oncastproc"] else val.lower() == "yes"
        
        for field, key, converter in [("cast point", "cast_point", float), ("cast backswing", "cast_backswing", float)]:
            val = self._extract_infobox_value(rf'\|\s*{field}\s*=\s*([\d.]+)', ability_text)
            if val:
                ability[key] = converter(val)
        
        cast_immediate = self._extract_infobox_value(r'\|\s*cast immediate\s*=\s*(.+)', ability_text)
        if cast_immediate:
            ability["cast_immediate"] = cast_immediate.lower() == "true"
        
        for field, key in [("mana", "mana"), ("cooldown", "cooldown"), ("cooldown aghs", "cooldown_aghs")]:
            val = self._extract_infobox_value(rf'\|\s*{field}\s*=\s*(.+)', ability_text)
            if val:
                val = val.strip()
                if '/' in val:
                    ability[key] = val
                else:
                    try:
                        ability[key] = float(val) if '.' in val else int(val)
                    except ValueError:
                        ability[key] = val
        
        traits = self._extract_traits(ability_text, ability_name)
        if traits:
            ability["traits"] = traits
        
        aghs_upgrade = self._extract_infobox_value(r'\|\s*aghanimsupgrade\s*=\s*(.+)', ability_text)
        if aghs_upgrade:
            ability["aghanims_scepter_upgrade"] = self._clean_wiki_markup(aghs_upgrade)
        
        aghs_shard = self._extract_infobox_value(r'\|\s*aghshard\s*=\s*(.+)', ability_text)
        if aghs_shard:
            ability["aghanims_shard_upgrade"] = self._clean_wiki_markup(aghs_shard)
        
        notes = self._extract_notes(ability_text)
        if notes:
            ability["notes"] = notes
        
        return ability
    
    def _extract_traits(self, ability_text: str, ability_name: str) -> Dict[str, str]:
        traits = {}
        
        trait_lines = re.findall(r'\|\s*trait(\d+)(?:\s+(\w+))?\s*=\s*(.+)', ability_text, re.MULTILINE)
        value_lines = re.findall(r'\|\s*value(\d+)(?:\s+(\w+))?\s*=\s*(.+)', ability_text, re.MULTILINE)
        
        trait_map = {}
        for trait_num, modifier, trait_name in trait_lines:
            key = (trait_num, modifier if modifier else "")
            trait_map[key] = trait_name.strip()
        
        value_map = {}
        for val_num, modifier, val_content in value_lines:
            key = (val_num, modifier if modifier else "")
            raw_value = val_content.strip()
            
            var_name = f"{ability_name} value{val_num}"
            if modifier:
                var_name += f" {modifier}"
            
            if re.match(r'^[\d./% -]+$', raw_value):
                self.evaluator.variables[var_name] = raw_value
                value_map[key] = raw_value
            else:
                evaluated = self.evaluator.evaluate(raw_value)
                evaluated = self._clean_wiki_markup(evaluated)
                self.evaluator.variables[var_name] = evaluated
                value_map[key] = evaluated
        
        for key, trait_name in trait_map.items():
            if key in value_map:
                trait_name_clean = self._clean_wiki_markup(trait_name)
                snake_key = trait_name_clean.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                if key[1]:
                    snake_key += f"_{key[1]}"
                traits[snake_key] = value_map[key]
        
        return traits
    
    def _extract_notes(self, ability_text: str) -> List[str]:
        notes_start_match = re.search(r'\|\s*notes\s*=\s*', ability_text)
        if not notes_start_match:
            return []
        
        notes_start = notes_start_match.end()
        remaining_text = ability_text[notes_start:]
        
        notes_end_match = re.search(r'\n\s*\|\s*(?![\*])[a-zA-Z][\w\s]*=', remaining_text)
        notes_text = remaining_text[:notes_end_match.start()] if notes_end_match else remaining_text
        notes_text = notes_text.strip()
        
        if not notes_text:
            return []
        
        return self._parse_notes_text(notes_text)
    
    def _parse_notes_text(self, notes_text: str) -> List[str]:
        # IMPORTANT: First evaluate the ENTIRE notes text to process any
        # {{#vardefine}} or other templates that set up variables for later use
        # This ensures vardefine at the start affects calcs later in the text
        evaluated_text = self.evaluator.evaluate(notes_text)
        
        notes = []
        lines = evaluated_text.split('\n')
        current_note = []
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                if current_note:
                    note_text = ' '.join(current_note)
                    note_text = self._clean_wiki_markup(note_text)
                    if note_text and len(note_text) > 5:
                        notes.append(note_text)
                    current_note = []
                continue
            
            if stripped.startswith('* ') and not stripped.startswith('** '):
                if current_note:
                    note_text = ' '.join(current_note)
                    note_text = self._clean_wiki_markup(note_text)
                    if note_text and len(note_text) > 5:
                        notes.append(note_text)
                    current_note = []
                current_note.append(stripped[2:])
            elif stripped.startswith('** '):
                current_note.append(stripped[3:])
            elif current_note:
                current_note.append(stripped)
        
        if current_note:
            note_text = ' '.join(current_note)
            note_text = self._clean_wiki_markup(note_text)
            if note_text and len(note_text) > 5:
                notes.append(note_text)
        
        return notes
    
    def _parse_abilities(self):
        ability_sections = [(name, content) for name, content in self.sections.items() if '{{Ability' in content]
        
        all_abilities = []
        for section_name, section_content in ability_sections:
            for ability_text in self._extract_ability_blocks(section_content):
                ability = self._parse_ability_block(ability_text)
                if ability.get("name"):
                    ability["section"] = section_name
                    all_abilities.append(ability)
        
        if all_abilities:
            innate = [a for a in all_abilities if a.get("section") == "Innate Abilities"]
            invoked = [a for a in all_abilities if a.get("section") == "Invoked Abilities"]
            regular = [a for a in all_abilities if a.get("section") not in ["Innate Abilities", "Invoked Abilities"]]
            
            for ability in all_abilities:
                ability.pop("section", None)
            
            if innate:
                self.parsed_data["innate_abilities"] = innate
            if invoked:
                self.parsed_data["invoked_abilities"] = invoked
            if regular:
                self.parsed_data["abilities"] = regular
    
    def _parse_trivia(self):
        trivia_section = self.sections.get("Trivia", "")
        if not trivia_section:
            return
        
        trivia_items = []
        for item in re.split(r'\n\s*\*\s*', trivia_section):
            item = item.strip()
            if item and not item.startswith('{{') and not item.startswith('=='):
                item = self._clean_wiki_markup(item)
                item = re.sub(r'\n\s*\*\*.*', '', item).strip()
                if item and len(item) > 20:
                    trivia_items.append(item)
        
        if trivia_items:
            self.parsed_data["trivia"] = trivia_items
    
    def to_json(self, filepath: str = None, indent: int = 2) -> str:
        json_str = json.dumps(self.parsed_data, indent=indent, ensure_ascii=False)
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
        return json_str
    
    def inject_guide_data(self, guide_data: Dict[str, Any]):
        if guide_data:
            self.parsed_data["hero_guide"] = guide_data
    
    def inject_talent_data(self, talent_data: Dict[str, Any]):
        if talent_data:
            self.parsed_data["talent_tree"] = talent_data
    
    def inject_changelog_data(self, changelog_data: Dict[str, Any]):
        if changelog_data:
            self.parsed_data["changelog"] = changelog_data


def parse_hero_wiki(wiki_text: str, excluded_sections: List[str] = None) -> Dict[str, Any]:
    parser = EnhancedHeroWikiParser(wiki_text, excluded_sections)
    return parser.parse()


HeroWikiParser = EnhancedHeroWikiParser

# TALENT PARSER

def clean_wiki_text_simple(text: str) -> str:
    """Simple wiki text cleaner for talent/guide/changelog parsers."""
    if not text:
        return ""
    
    # Remove {{H|Hero Name}} -> Hero Name
    text = re.sub(r'\{\{H\|([^}|]+)(?:\|[^}]+)?\}\}', r'\1', text)
    
    # Remove {{A|Ability|Hero}} -> Ability
    text = re.sub(r'\{\{A\|([^}|]+)(?:\|[^}]+)?\}\}', r'\1', text)
    
    # Remove {{I|Item|...}} -> Item
    text = re.sub(r'\{\{I\|([^}|]+)(?:\|[^}]+)?\}\}', r'\1', text)
    
    # Remove {{Attribute ID|strength}} -> strength
    text = re.sub(r'\{\{Attribute ID\|([^}]+)\}\}', r'\1', text)
    
    # Remove [[Link|Text]] -> Text or [[Link]] -> Link
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    
    # Remove {{tooltip|Text|Hover}} -> Text
    text = re.sub(r'\{\{tooltip\|([^}|]+)\|[^}]+\}\}', r'\1', text)
    
    # Remove {{cf|...}} and {{Symbol|...}}
    text = re.sub(r'\{\{cf\|[^}]+\}\}\s*', '', text)
    text = re.sub(r'\{\{Symbol\|([^}]+)\}\}', r'\1', text)
    
    # Remove bolding and italics
    text = text.replace("'''", "")
    text = text.replace("''", "")
    
    return text.strip()


def parse_wiki_talent_text(wiki_text: str) -> Dict[str, Any]:
    """
    Parses raw Dota 2 Wiki syntax for talents into a dictionary.
    """
    # Extract Hero Name
    hero_match = re.search(r'\|\s*hero\s*=\s*(.*)', wiki_text, re.IGNORECASE)
    hero_name = hero_match.group(1).strip() if hero_match else "Unknown"

    # Extract all key-value pairs
    raw_data = {}
    matches = re.findall(r'\|\s*([tv][1-4](?:left|right))\s*=\s*(.*)', wiki_text)
    for key, val in matches:
        raw_data[key] = val.strip()

    def format_talent_string(text, value):
        if not text:
            return "Unknown Talent"
        
        # Clean Wiki Templates
        text = re.sub(r'\{\{A\|([^|]+)(?:\|[^}]+)?\}\}', r'\1', text)

        if 's:values' in text:
            text = text.replace('s:values', f"{value}s")
        elif 's:value' in text:
            text = text.replace('s:value', value)
        else:
            formatted_text = text.title()
            text = f"+{value} {formatted_text}"
            
        return text

    # Map Wiki Levels (1-4) to Game Levels (10-25)
    talents = {}
    level_map = {'1': '10', '2': '15', '3': '20', '4': '25'}

    for i in range(1, 5):
        idx = str(i)
        game_level = level_map[idx]
        
        t_left_raw = raw_data.get(f't{idx}left', '')
        v_left_raw = raw_data.get(f'v{idx}left', '')
        
        t_right_raw = raw_data.get(f't{idx}right', '')
        v_right_raw = raw_data.get(f'v{idx}right', '')

        left_final = format_talent_string(t_left_raw, v_left_raw)
        right_final = format_talent_string(t_right_raw, v_right_raw)

        talents[f"level_{game_level}"] = [left_final, right_final]

    return {
        "hero": hero_name,
        "talents": talents
    }


# GUIDE PARSER

def parse_bullet_points(raw_text: str) -> List[str]:
    """Extracts bullet points starting with * from a block of text."""
    items = []
    if not raw_text:
        return items

    for line in raw_text.split('\n'):
        if line.strip().startswith('*'):
            clean_line = line.strip().lstrip('*').strip()
            items.append(clean_wiki_text_simple(clean_line))
    return items


def parse_dota_wiki_guide(wiki_text: str) -> Dict[str, Any]:
    """
    Parses Dota 2 Wiki guide syntax into a dictionary.
    """
    result = {}

    # 1. PARSE GAMEPLAY GUIDE (Intro, Pros, Cons)
    start_index = wiki_text.find("{{hero intro")
    
    if start_index != -1:
        end_index = wiki_text.find("==", start_index)
        if end_index == -1:
            end_index = len(wiki_text)
            
        intro_content = wiki_text[start_index:end_index]

        intro_match = re.search(r'\|\s*intro\s*=\s*(.*?)(?=\n\s*\|)', intro_content, re.DOTALL)
        intro_text = clean_wiki_text_simple(intro_match.group(1)) if intro_match else ""

        pros_match = re.search(r'\|\s*pros\s*=\s*(.*?)(?=\n\s*\|)', intro_content, re.DOTALL)
        pros_list = parse_bullet_points(pros_match.group(1)) if pros_match else []

        cons_match = re.search(r'\|\s*cons\s*=\s*(.*?)(?=\n\s*\}\})', intro_content, re.DOTALL)
        cons_list = parse_bullet_points(cons_match.group(1)) if cons_match else []

        result["gameplay_guide"] = {
            "intro": intro_text,
            "pros": pros_list,
            "cons": cons_list
        }

    # 2. PARSE ABILITY BUILDS
    builds = []
    build_matches = re.findall(r'\{\{build order\s*\|(.*?)\}\}', wiki_text, re.DOTALL)
    
    for build_content in build_matches:
        build_obj = {"title": "Unknown", "progression": {}}
        
        title_match = re.search(r'title\s*=\s*(.*)', build_content)
        if title_match:
            build_obj["title"] = title_match.group(1).strip()
        
        lvl_matches = re.findall(r'\|\s*lvl(\d+)\s*=\s*(.*)', build_content)
        for lvl, ability in lvl_matches:
            build_obj["progression"][lvl] = ability.strip()
            
        builds.append(build_obj)
    
    result["ability_builds"] = builds

    # 3. PARSE TIPS & TACTICS
    result["tips"] = {"general": [], "abilities": {}}
    
    tips_section_match = re.search(r'== Tips & Tactics ==(.*?)(?=== Items|== Talents|$)', wiki_text, re.DOTALL)
    if tips_section_match:
        tips_content = tips_section_match.group(1)
        
        parts = re.split(r'===\s*Abilities\s*===', tips_content)
        
        if len(parts) > 0:
            general_raw = parts[0]
            if "=== General ===" in general_raw:
                general_raw = general_raw.split("=== General ===")[1]
            result["tips"]["general"] = parse_bullet_points(general_raw)

        if len(parts) > 1:
            abilities_raw = parts[1]
            ability_sections = re.split(r'====\s*\{\{A\|([^|]+).*?\}\}\s*====', abilities_raw)
            
            for i in range(1, len(ability_sections), 2):
                a_name = ability_sections[i].strip()
                a_text = ability_sections[i+1]
                result["tips"]["abilities"][a_name] = parse_bullet_points(a_text)

    # 4. PARSE ITEMS
    result["recommended_items"] = {}
    
    items_match = re.search(r'<section begin=Items />(.*?)<section end=Items />', wiki_text, re.DOTALL)
    
    if items_match:
        items_content = items_match.group(1)
        
        key_map = {
            "Starting items": "starting_items",
            "Early game": "early_game",
            "Mid game": "mid_game",
            "Late game": "late_game",
            "Situational items": "situational_items"
        }
        
        chunks = re.split(r"'''(.*?):'''", items_content)
        
        for i in range(1, len(chunks), 2):
            header_raw = chunks[i].strip()
            content_raw = chunks[i+1]
            
            json_key = key_map.get(header_raw, header_raw.lower().replace(" ", "_"))
            
            items_list = []
            item_lines = content_raw.strip().split('\n')
            for line in item_lines:
                if not line.strip().startswith('*'):
                    continue
                
                item_name_match = re.search(r'\{\{I\|([^|]+)', line)
                if not item_name_match:
                    continue
                
                item_name = item_name_match.group(1)
                
                desc_text = re.sub(r'\*\s*\{\{I\|[^}]+\}\}', '', line).strip()
                desc_text = clean_wiki_text_simple(desc_text)
                
                items_list.append({
                    "item": item_name,
                    "description": desc_text
                })
            
            if items_list:
                result["recommended_items"][json_key] = items_list

    return result


# CHANGELOG PARSER

def parse_talent_changes(content: str) -> tuple:
    """Extracts and formats {{Tal change}} blocks."""
    formatted_changes = []
    
    match = re.search(r'\{\{Tal change\s*\|(.*?)\}\}', content, re.DOTALL | re.IGNORECASE)
    
    if match:
        raw_params = match.group(1)
        lvl_map = {'1': '10', '2': '15', '3': '20', '4': '25'}
        side_map = {'l': 'Left', 'r': 'Right'}
        
        params = raw_params.split('|')
        
        for param in params:
            p_match = re.search(r't([1-4])([lr])\s*=\s*(.*)', param.strip(), re.IGNORECASE)
            if p_match:
                tier = p_match.group(1)
                side = p_match.group(2)
                val = p_match.group(3)
                
                val = val.replace(';c', '').strip()
                val = clean_wiki_text_simple(val)
                
                level = lvl_map.get(tier, '?')
                branch = side_map.get(side.lower(), '?')
                
                formatted_changes.append(f"Level {level} {branch} Branch: {val}")
        
        content = content.replace(match.group(0), "")
        
    return formatted_changes, content


def parse_version_history(wiki_text: str) -> Dict[str, Any]:
    """
    Parses Dota 2 Wiki version history into a dictionary.
    """
    data = {
        "the_most_recent": {},
        "previous_versions": []
    }
    
    chunks = wiki_text.split('{{VersionTableElement|')
    
    versions_found = []
    target_chunks = chunks[1:4]  # Get first 3 version elements
    
    for chunk in target_chunks:
        if '|' not in chunk:
            continue
        
        version_num, remaining = chunk.split('|', 1)
        version_num = version_num.strip()
        
        talent_changes, remaining = parse_talent_changes(remaining)
        
        changes_list = []
        
        lines = remaining.split('\n')
        for line in lines:
            clean_line = line.strip()
            
            if '{{VersionTableEnd}}' in clean_line:
                break
            if clean_line == '}}':
                continue
            
            if clean_line.startswith('*'):
                indent = ""
                content = clean_line
                
                if clean_line.startswith('***'):
                    indent = "    * "
                    content = clean_line.lstrip('*').strip()
                elif clean_line.startswith('**'):
                    indent = "  * "
                    content = clean_line.lstrip('*').strip()
                else:
                    content = clean_line.lstrip('*').strip()
                
                final_text = clean_wiki_text_simple(content)
                if final_text:
                    changes_list.append(f"{indent}{final_text}")
                    
            elif clean_line.startswith(':'):
                indent = "  * "
                if clean_line.startswith(':::'):
                    indent = "      "
                elif clean_line.startswith('::'):
                    indent = "    "
                
                content = clean_line.lstrip(':').strip()
                final_text = clean_wiki_text_simple(content)
                if final_text:
                    changes_list.append(f"{indent}{final_text}")

        changes_list.extend(talent_changes)
        
        version_obj = {
            "version": version_num,
            "changes": changes_list
        }
        versions_found.append(version_obj)

    if versions_found:
        data["the_most_recent"] = versions_found[0]
        data["previous_versions"] = versions_found[1:]
        
    return data


# COMPREHENSIVE PARSER

def parse_hero_complete(
    hero_wiki_text: str,
    talent_wiki_text: str = None,
    guide_wiki_text: str = None,
    changelog_wiki_text: str = None,
    excluded_sections: List[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive hero parser that combines all wiki data sources.
    """
    # Parse main hero data
    parser = EnhancedHeroWikiParser(hero_wiki_text, excluded_sections)
    result = parser.parse()
    
    # Inject talent data if provided
    if talent_wiki_text:
        try:
            talent_data = parse_wiki_talent_text(talent_wiki_text)
            if talent_data and talent_data.get("talents"):
                result["talent_tree"] = talent_data["talents"]
        except Exception as e:
            result["_talent_parse_error"] = str(e)
    
    # Inject guide data if provided
    if guide_wiki_text:
        try:
            guide_data = parse_dota_wiki_guide(guide_wiki_text)
            if guide_data:
                result["hero_guide"] = guide_data
        except Exception as e:
            result["_guide_parse_error"] = str(e)
    
    # Inject changelog data if provided
    if changelog_wiki_text:
        try:
            changelog_data = parse_version_history(changelog_wiki_text)
            if changelog_data:
                result["changelog"] = changelog_data
        except Exception as e:
            result["_changelog_parse_error"] = str(e)
    
    return result


def parse_hero_complete_to_json(
    hero_wiki_text: str,
    talent_wiki_text: str = None,
    guide_wiki_text: str = None,
    changelog_wiki_text: str = None,
    excluded_sections: List[str] = None,
    indent: int = 2
) -> str:
    """
    Comprehensive hero parser that returns JSON string.
    """
    result = parse_hero_complete(
        hero_wiki_text,
        talent_wiki_text,
        guide_wiki_text,
        changelog_wiki_text,
        excluded_sections
    )
    return json.dumps(result, indent=indent, ensure_ascii=False)


# FETCH AND PARSE FROM WIKI

def fetch_and_parse_hero(
    hero_name: str,
    include_guide: bool = True,
    include_talents: bool = True,
    include_changelog: bool = True,
    excluded_sections: List[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Fetches all wiki pages for a hero and parses them into a complete data structure.
    """
    # Fetch main hero page
    if verbose:
        print(f"Fetching {hero_name}...")
    hero_wiki_text = fetch_raw_wikitext(hero_name)
    
    if not hero_wiki_text:
        raise ValueError(f"Could not fetch wiki page for hero: {hero_name}")
    
    # Fetch optional pages
    talent_wiki_text = None
    guide_wiki_text = None
    changelog_wiki_text = None
    
    if include_talents:
        if verbose:
            print(f"Fetching {hero_name}/Talents...")
        talent_wiki_text = fetch_raw_wikitext(f"{hero_name}/Talents")
    
    if include_guide:
        if verbose:
            print(f"Fetching {hero_name}/Guide...")
        guide_wiki_text = fetch_raw_wikitext(f"{hero_name}/Guide")
    
    if include_changelog:
        if verbose:
            print(f"Fetching {hero_name}/Changelogs...")
        changelog_wiki_text = fetch_raw_wikitext(f"{hero_name}/Changelogs")
    
    if verbose:
        print("Parsing...")
    
    # Parse everything
    result = parse_hero_complete(
        hero_wiki_text,
        talent_wiki_text,
        guide_wiki_text,
        changelog_wiki_text,
        excluded_sections
    )
    
    if verbose:
        print("Done!")
    
    return result


def fetch_and_parse_hero_to_json(
    hero_name: str,
    include_guide: bool = True,
    include_talents: bool = True,
    include_changelog: bool = True,
    excluded_sections: List[str] = None,
    indent: int = 2,
    verbose: bool = False
) -> str:
    """
    Fetches all wiki pages for a hero and returns parsed data as JSON string.
    """
    result = fetch_and_parse_hero(
        hero_name,
        include_guide,
        include_talents,
        include_changelog,
        excluded_sections,
        verbose
    )
    return json.dumps(result, indent=indent, ensure_ascii=False)


def fetch_and_save_hero(
    hero_name: str,
    output_path: str = None,
    include_guide: bool = True,
    include_talents: bool = True,
    include_changelog: bool = True,
    excluded_sections: List[str] = None,
    indent: int = 2,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Fetches all wiki pages for a hero, parses them, and saves to a JSON file.
    """
    result = fetch_and_parse_hero(
        hero_name,
        include_guide,
        include_talents,
        include_changelog,
        excluded_sections,
        verbose
    )
    
    # Generate output path if not provided
    if output_path is None:
        safe_name = hero_name.lower().replace(" ", "_").replace("'", "")
        output_path = f"{safe_name}_data.json"
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=indent, ensure_ascii=False)
    
    if verbose:
        print(f"Saved to {output_path}")
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        sys.exit(1)
    
    hero_name = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        fetch_and_save_hero(hero_name, output_file, verbose=True)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)