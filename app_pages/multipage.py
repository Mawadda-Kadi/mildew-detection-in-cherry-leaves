import streamlit as st


# Class to generate multiple Streamlit pages using an object oriented approach
class MultiPage:

    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,)

    def add_page(self, title, func) -> None:
        self.pages.append({"title": title, "function": func})

    def _inject_custom_css(self):
        css = """
        <style>
        h1 { font-size: 28px; }
        h2 { font-size: 22px; }
        h3 { font-size: 20px; }
        h4 { font-size: 18px; }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

    def run(self):
        self._inject_custom_css()
        st.title(self.app_name)
        page = st.sidebar.radio('Menu', self.pages, format_func=lambda page: page['title'])
        page['function']()