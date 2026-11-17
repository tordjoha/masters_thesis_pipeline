import tree_sitter_java
from tree_sitter import Language, Parser
from config import MAX_PROMPT_TOKENS
from generator import count_tokens

java_language = Language(tree_sitter_java.language())
parser = Parser(java_language)


def node_text(source_bytes, start_byte, end_byte):
    """Extract node text from the raw byte buffer. Tree-sitter offsets are
    BYTE offsets, so slicing must happen on bytes and decode per slice --
    slicing a decoded str is only coincidentally correct for pure-ASCII
    files, and older Apache Commons sources contain Latin-1 bytes (e.g.
    0xA9 in license headers) that break that assumption."""
    return source_bytes[start_byte:end_byte].decode('utf-8', errors='replace')


def parse_java_file(filepath):
    """Parse a Java file, returning (tree, source_bytes). All extraction
    functions slice source_bytes via node_text."""
    with open(filepath, 'rb') as f:
        data = f.read()
    tree = parser.parse(data)
    return tree, data


def extract_imports(tree, source):
    imports = []
    for node in tree.root_node.children:
        if node.type == 'import_declaration':
            imports.append(node_text(source, node.start_byte, node.end_byte))
    return imports


def extract_class_method_names(tree, source):
    """Get all method names actually declared in this class."""
    declared = set()
    for node in tree.root_node.children:
        if node.type == 'class_declaration':
            class_body = next(
                (c for c in node.children if c.type == 'class_body'),
                None
            )
            if not class_body:
                continue
            for member in class_body.children:
                if member.type == 'method_declaration':
                    method_name = next(
                        (c for c in member.children if c.type == 'identifier'),
                        None
                    )
                    if method_name:
                        declared.add(node_text(source, method_name.start_byte, method_name.end_byte))
    return declared


def extract_called_methods(tree, source, target_method_name):
    """Find method names called within the target method body."""
    called = set()

    for node in tree.root_node.children:
        if node.type == 'class_declaration':
            class_body = next(
                (c for c in node.children if c.type == 'class_body'),
                None
            )
            if not class_body:
                continue

            for member in class_body.children:
                if member.type == 'method_declaration':
                    method_name = next(
                        (c for c in member.children if c.type == 'identifier'),
                        None
                    )
                    if method_name and node_text(source, method_name.start_byte, method_name.end_byte) == target_method_name:
                        def walk(node):
                            if node.type == 'method_invocation':
                                name = next(
                                    (c for c in node.children if c.type == 'identifier'),
                                    None
                                )
                                if name:
                                    called.add(node_text(source, name.start_byte, name.end_byte))
                            for child in node.children:
                                walk(child)
                        walk(member)
    return called


def extract_class_components(tree, source, target_method_name):
    components = {
        'fields': [],
        'constructors': [],
        'target_method': None,
        'other_methods': []
    }

    # find sibling methods actually called by the target
    called_methods = extract_called_methods(tree, source, target_method_name)
    declared_methods = extract_class_method_names(tree, source)
    relevant_methods = (called_methods & declared_methods) - {target_method_name}

    for node in tree.root_node.children:
        if node.type == 'class_declaration':
            class_body = next(
                (c for c in node.children if c.type == 'class_body'),
                None
            )
            if not class_body:
                continue

            for member in class_body.children:
                if member.type == 'field_declaration':
                    components['fields'].append(
                        node_text(source, member.start_byte, member.end_byte)
                    )

                elif member.type == 'constructor_declaration':
                    constructor_body = next(
                        (c for c in member.children if c.type == 'constructor_body'),
                        None
                    )
                    if constructor_body:
                        sig_text = node_text(source, member.start_byte, constructor_body.start_byte).strip()
                        components['constructors'].append(sig_text + ' {...}')

                elif member.type == 'method_declaration':
                    method_name = next(
                        (c for c in member.children if c.type == 'identifier'),
                        None
                    )
                    if not method_name:
                        continue

                    name_str = node_text(source, method_name.start_byte, method_name.end_byte)

                    if name_str == target_method_name:
                        components['target_method'] = node_text(source, member.start_byte, member.end_byte)
                    elif name_str in relevant_methods:
                        block = next(
                            (c for c in member.children if c.type == 'block'),
                            None
                        )
                        if block:
                            sig_text = node_text(source, member.start_byte, block.start_byte).strip()
                            components['other_methods'].append(sig_text + ' {...}')

    return components


def assemble_segment_prompt(segments, modified_class):
    """
    Assemble a prompt from the given segments.
    Same prompt template for generation and coalition masking
    """
    class_name = modified_class.split('.')[-1]
    instruction = (
        f"Generate a single JUnit 4 test method for the target method above.\n"
        f"The class under test is {class_name}. Use the constructor directly: new {class_name}(...).\n"
        f"Only use methods and fields shown in the segments above.\n"
        f"Do not include import statements, class declarations, or package declarations.\n"
        f"Start your response directly with @Test and end after the closing brace of the test method."
    )
    return (
        f"[IMPORTS]\n{segments['imports']}\n[/IMPORTS]\n\n"
        f"[FIELDS]\n{segments['fields']}\n[/FIELDS]\n\n"
        f"[CONSTRUCTORS]\n{segments['constructors']}\n[/CONSTRUCTORS]\n\n"
        f"[OTHER_METHODS]\n{segments['other_methods']}\n[/OTHER_METHODS]\n\n"
        f"[TARGET_METHOD]\n{segments['target_method']}\n[/TARGET_METHOD]\n\n"
        f"{instruction}"
    )


def build_prompt(filepath, target_method_name, modified_class):
    """
    Build a segmented prompt for the target method.
    Returns (prompt, components, segments, exclusion_reason).
    segments dict maps segment name to its text content for use in SHAP masking.
    exclusion_reason is None if the prompt is within the token cap,
    otherwise a string explaining why the bug was excluded.
    """
    tree, source = parse_java_file(filepath)
    imports = extract_imports(tree, source)
    components = extract_class_components(tree, source, target_method_name)

    if components['target_method'] is None:
        return None, components, None, f"target method '{target_method_name}' not found"

    # store raw segment content for SHAP masking later
    segments = {
        'imports': '\n'.join(imports),
        'fields': '\n'.join(components['fields']) if components['fields'] else 'none',
        'constructors': '\n'.join(components['constructors']) if components['constructors'] else 'none',
        'other_methods': '\n'.join(components['other_methods']) if components['other_methods'] else 'none',
        'target_method': components['target_method']
    }

    def assemble_prompt(segs):
        return assemble_segment_prompt(segs, modified_class)

    # always define minimal prompt first - constructors kept as essential context
    minimal_segments = {
        'imports': segments['imports'],
        'fields': 'none',
        'constructors': segments['constructors'],
        'other_methods': 'none',
        'target_method': segments['target_method']
    }

    minimal_prompt = assemble_prompt(minimal_segments)

    if count_tokens(minimal_prompt) > MAX_PROMPT_TOKENS:
        return None, components, None, f"minimal prompt exceeds {MAX_PROMPT_TOKENS} token cap"

    # try full prompt first
    full_prompt = assemble_prompt(segments)
    if count_tokens(full_prompt) <= MAX_PROMPT_TOKENS:
        return full_prompt, components, segments, None

    # trim fields first
    no_fields_segments = {**segments, 'fields': 'none'}
    no_fields_prompt = assemble_prompt(no_fields_segments)
    if count_tokens(no_fields_prompt) <= MAX_PROMPT_TOKENS:
        return no_fields_prompt, components, no_fields_segments, None

    # fall back to minimal
    return minimal_prompt, components, minimal_segments, None
