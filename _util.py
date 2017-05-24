
def click_nowrap():
    """Disable all wrapping, so that click doesn't mangle good docstrings."""
    from textwrap import indent

    import click.formatting

    def _wrap_text(text, width=None, initial_indent='', subsequent_indent='',
                   **kwargs):
        # Indent all but the first line
        def predicate(line):
            first = getattr(predicate, 'first', True)
            predicate.first = False
            return not first

        result = indent(text, subsequent_indent, predicate)
        return initial_indent + result

    # Overwrite the default method
    click.formatting.wrap_text = _wrap_text
