from sparknlp.base import Annotation
from sparknlp_display import RelationExtractionVisualizer
import ipywidgets as widgets
from IPython.display import display

index = 0

output = widgets.Output()
btn_next = widgets.Button(description="→")
btn_prev = widgets.Button(description="←")
btn_first = widgets.Button(description="↺")

vis = RelationExtractionVisualizer()

def build_mock(text):
    words = text.split()
    
    # Build DOCUMENT
    document = [Annotation("DOCUMENT", 0, len(text) - 1, text, {}, None)]
    
    # Build TOKENS with char offsets
    tokens = []
    cursor = 0
    for w in words:
        start = text.find(w, cursor)
        end = start + len(w) - 1
        tokens.append(Annotation("TOKEN", start, end, w, {}, None))
        cursor = end + 1

    # Assume simple relation: first word -> last word
    relations = [
        Annotation(
            "RELATION",
            0, len(text) - 1,
            "love",
            {
                "entity1": "Person",
                "entity1_begin": str(tokens[0].begin),
                "entity1_end": str(tokens[0].end),
                "chunk1": tokens[0].result,
                "entity2": "Object",
                "entity2_begin": str(tokens[-1].begin),
                "entity2_end": str(tokens[-1].end),
                "chunk2": tokens[-1].result
            },
            None
        )
    ]

    return {
        "document": document,
        "tokens": tokens,
        "relations": relations
    }

def render(examples):
    output.clear_output()
    mock_result = build_mock(examples[index])

    with output:
        vis.display(
            mock_result,
            relation_col='relations',
            document_col='document',
            show_relations=True
        )
        print(f"{index+1} / {len(examples)}")


def next_click(b):
    global index
    index = (index + 1) % len(examples)
    render()


def prev_click(b):
    global index
    index = (index - 1) % len(examples)
    render()


btn_next.on_click(next_click)
btn_prev.on_click(prev_click)
btn_first.on_click(first_click)