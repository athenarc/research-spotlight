from spacy import displacy
import ipywidgets as widgets
from IPython.display import display, HTML

index = 0
examples = []

output = widgets.Output()
btn_next = widgets.Button(description="→")
btn_prev = widgets.Button(description="←")
btn_first = widgets.Button(description="↺")


def render(data=None):
    global index, examples

    if data is not None:
        examples = data

    output.clear_output()

    if not examples:
        return

    item = examples[index]

    options = {"colors":{"METHOD":"#7aecec"}}

    html = displacy.render([item], style="ent", options=options, jupyter=False, page=False, manual=True)

    with output:
        display(HTML(html))
        print(f"{index+1} / {len(examples)}")

def first_click(b):
    global index
    index = 0
    render()
    
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
