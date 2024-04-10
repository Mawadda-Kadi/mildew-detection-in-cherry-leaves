import streamlit as st


class MultiPage:
    """Framework for combining multiple streamlit applications."""

    def __init__(self, app_name: str):
        self.pages = []
        self.app_name = app_name

    def add_page(self, title, func):
        """Adds pages to the app."""
        self.pages.append({
            "title": title,
            "function": func
        })

    def run(self):
        """Run the app with the sidebar to choose between pages."""
        st.sidebar.title(self.app_name)
        page = st.sidebar.radio(
            'App Navigation',
            self.pages,
            format_func=lambda page: page['title']
        )

        # run the app function
        page['function']()