import re

URL_PATTERN = re.compile(
    # https://stackoverflow.com/questions/3809401/what-is-a-good-regular-expression-to-match-a-url#answer-3809435
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
)


def preprocess_text(text: str, strip_urls: bool = False) -> str:
    '''
    Performs basic preprocessing on text data by converting it to lowercase,
    removing non-alphanumeric characters (including emoji) and collapsing
    whitespace. The function can optionally remove URLs from the text.

    Args:
        text: The text to preprocess. This is a single example.
        strip_urls: Whether to remove URLs from the text.

    Returns:
        The preprocessed text.
    '''

    text = text.lower()
    if strip_urls:
        text = re.sub(URL_PATTERN, ' ', text)
    text = re.sub(r'[\d\W\s]+', ' ', text)
    return text.strip()
