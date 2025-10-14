# Shims con los mismos NOMBRES que tu pipeline espera
# para que el pickle los encuentre y cargue sin error.

def clean_text(s: str) -> str:
    # No-op: ya limpias aguas arriba
    return s

def stemmed_tokenizer(text: str):
    # Tokenizador mínimo sin NLTK (evita descargas y dependencias)
    # Si quieres, aquí puedes meter tu tokenización custom ligera.
    return text.split()
