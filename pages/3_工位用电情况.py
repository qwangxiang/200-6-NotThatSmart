import streamlit as st
from streamlit_extras.card import card

# 初始化 session state
if 'data' not in st.session_state:
    st.session_state.data = 'initial'

def change_data(value):
    def callback():
        st.session_state.data = value
    return callback

def Reset():
    st.session_state.data = 'initial'

if __name__ == '__main__':
    st.title('工位用电情况')

    # 放一个简单的card
    if st.session_state.data == 'initial':
        col1, col2 = st.columns([1, 1])
        with col1:
            card(
                title='点击有惊喜',
                text='initial',
                on_click=change_data('click1')
            )
        with col2:
            card(
                title='点击有惊喜',
                text='changed',
                on_click=change_data('click2')
            )
    # 根据 session state 的值展示不同的内容
    else:
        card(
            title='展示data',
            text=str(st.session_state.data),
            on_click=Reset
        )

