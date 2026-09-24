class Trie:
    def __init__(self, trie_dict=None):
        self.trie_dict = trie_dict or {}

    @classmethod
    def load_from_dict(cls, data):
        obj = cls()
        obj.trie_dict = data
        return obj

    def get(self, tokens):
        node = self.trie_dict
        for token in tokens:
            if token not in node:
                return []
            node = node[token]
        if isinstance(node, dict):
            return list(node.keys())
        return node or []
