import pytest
from src.runtime_context import active_customer


@pytest.fixture(autouse=True)
def customer_scope():
    import streamlit as st
    st.cache_resource.clear()
    token = active_customer.set('customer_a')
    yield
    active_customer.reset(token)
