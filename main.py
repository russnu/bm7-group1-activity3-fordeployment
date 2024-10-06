import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# import io

st.markdown("# **Group Activity 2**")
st.markdown("""
---
**BM7 - Group 1**
* Segador, John Russel C.
* Tejada, Kurt Dhaniel A.
* Agor, John Schlieden A.
---
""")

# ======================================================================== #

st.markdown("## **Describing the dataset**")

# if 'dataset' not in st.session_state:
#     @st.cache_data
#     def load_data(url):
#         df = pd.read_csv(url)
#         return df

#     st.session_state.dataset = load_data('out.csv')
    
# df = st.session_state.dataset

# ======================================================================== #

# st.markdown("### Dataframe (Sampled)")
# sample_df = df.sample(n=100)
# st.dataframe(sample_df)

# ====================================== #
st.markdown("---")
st.markdown("### df.info()")

# buffer = io.StringIO()
# df.info(buf=buffer)
# s = buffer.getvalue()
st.text("""<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2500000 entries, 0 to 2499999
Data columns (total 18 columns):
 #   Column              Dtype  
---  ------              -----  
 0   url                 object 
 1   source              object 
 2   label               object 
 3   url_length          int64  
 4   starts_with_ip      bool   
 5   url_entropy         float64
 6   has_punycode        bool   
 7   digit_letter_ratio  float64
 8   dot_count           int64  
 9   at_count            int64  
 10  dash_count          int64  
 11  tld_count           int64  
 12  domain_has_digits   bool   
 13  subdomain_count     int64  
 14  nan_char_entropy    float64
 15  has_internal_links  bool   
 16  whois_data          object 
 17  domain_age_days     float64
dtypes: bool(4), float64(4), int64(6), object(4)
memory usage: 276.6+ MB""")

# ====================================== #
st.markdown("---")
st.markdown("### df.isna().sum()")

data = {
    "Feature": [
        "url", "source", "label", "url_length", "starts_with_ip", "url_entropy", 
        "has_punycode", "digit_letter_ratio", "dot_count", "at_count", "dash_count", 
        "tld_count", "domain_has_digits", "subdomain_count", "nan_char_entropy", 
        "has_internal_links", "whois_data", "domain_age_days"
    ],
    "Null Values": [
        1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, "545,300", "750,689"
    ]
}


df = pd.DataFrame(data)
df.set_index('Feature', inplace=True)

st.dataframe(df.style.hide(axis="index"))

# ====================================== #
st.markdown("---")
st.markdown("### df.describe()")
# st.dataframe(df.describe())

data = {
    'url_length': [2.500000e+06, 4.588017e+01, 7.439959e+01, 4.000000e+00, 1.700000e+01, 2.900000e+01, 5.200000e+01, 2.552300e+04],
    'url_entropy': [2.500000e+06, 3.907981e+00, 6.357209e-01, -0.000000e+00, 3.452820e+00, 3.911860e+00, 4.329283e+00, 6.048781e+00],
    'digit_letter_ratio': [2.499999e+06, 1.168567e-01, 2.451448e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.333333e-01, 2.084000e+01],
    'dot_count': [2.500000e+06, 2.174582e+00, 1.736380e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 2.110000e+02],
    'tld_count': [2.500000e+06, 3.923280e-02, 3.905095e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 6.500000e+01],
    'subdomain_count': [2.500000e+06, 7.777008e-01, 1.103257e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00, 4.300000e+01],
    'nan_char_entropy': [2.500000e+06, 4.651602e-01, 1.880023e-01, 0.000000e+00, 3.063967e-01, 4.154523e-01, 6.184690e-01, 1.901504e+00],
    'domain_age_days': [1.749311e+06, 4.863090e+03, 3.345879e+03, -8.600000e+01, 2.009000e+03, 4.281000e+03, 7.740000e+03, 4.554100e+04]
}

index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

df = pd.DataFrame(data, index=index)

st.dataframe(df)


# ====================================== #
st.markdown("---")
st.markdown("### Count of legitimate and phishing URLs")


# ====================================== #
st.markdown("---")
st.markdown("### Count of URLs based on sources")


# ====================================== #
st.markdown("---")
st.markdown("### Count of URLs based on whether they start with an IP address or not")


# ====================================== #
st.markdown("---")
st.markdown("### Count of URLs based on whether they have punycode characters")
# has_punycode = df['has_punycode'].value_counts()
# st.dataframe(has_punycode)

data = {
    "has_punycode": ["False", "True"],
    "count": [2497892, 2108]
}


df = pd.DataFrame(data)
df.set_index('has_punycode', inplace=True)

st.dataframe(df)

# ====================================== #
st.markdown("---")
st.markdown("### Count of URLs based on whether their domain contains digits")
# domain_has_digits = df['domain_has_digits'].value_counts()
# st.dataframe(domain_has_digits)

data = {
    "domain_has_digits": ["False", "True"],
    "count": [2226506, 273494]
}

df = pd.DataFrame(data)
df.set_index('domain_has_digits', inplace=True)
st.dataframe(df)

# ====================================== #
st.markdown("---")
st.markdown("### Count of URLs based on whether their subdirectory contains links")
# has_internal_links = df['has_internal_links'].value_counts()
# st.dataframe(has_internal_links)

data = {
    "has_internal_links": ["False", "True"],
    "count": [2440642, 59358]
}

df = pd.DataFrame(data)
df.set_index('has_internal_links', inplace=True)

st.dataframe(df)

# ================================================================================================================================== #
st.markdown("---")
st.markdown("## **Graphs**")
# ================================================================================================================================== #
st.markdown("---")
st.markdown("### **Segador**")

# starts_with_ip = df['starts_with_ip'].value_counts()
# colors = ['cornflowerblue', 'darkgrey']
# plt.pie(starts_with_ip,  labels = ['Does not start with IP Address', 'Starts with IP Address'], autopct='%1.1f%%', colors=colors)
# plt.title('Proportion of URLs Starting with IP Address')
# st.pyplot(plt)
# plt.clf()
st.image("segadorgraph1.png")

st.markdown("""The pie chart shows the proportions of URLs that start with an IP address and those 
            that do not. The URLs with an IP as their base account for ***98.9%*** of the dataset.""")



# url_length_average = df.groupby('label')['url_length'].mean().reset_index()

# plt.figure(figsize=(7, 7))
# colors = ['cornflowerblue', 'salmon']
# ax = sns.barplot(x='label', y='url_length', hue='label', data=url_length_average, palette=colors)

# plt.title('Average URL Length by Label (Phishing vs Legitimate)')
# plt.xlabel('Label')
# plt.ylabel('Average URL Length')

# for p in ax.patches:
#     ax.annotate(f'Average: {p.get_height():.1f}',(p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',  xytext=(0, 10), textcoords='offset points')
# st.pyplot(plt)
# plt.clf()
st.image("segadorgraph2.png")

st.markdown("""The diagram illustrates the average URL length of the legitimate URLs and phishing 
            URLs. It shows that phishing URLs have higher average lengths with ***71.9*** characters 
            compared to legitimate URLS with ***19.9***.""")

# ================================================================================================================================== #

st.markdown("---")
st.markdown("### **Tejada**")

# domain_has_digits = df['domain_has_digits'].value_counts()
# colors = ['salmon', 'skyblue']
# plt.pie(domain_has_digits,  labels = ['Domains that does not contain digits', 'Domains that contains digits'], autopct='%1.1f%%', colors=colors)
# plt.title('Proportion of Domains Containing Digits')
# st.pyplot(plt)
# plt.clf()
st.image("tejadagraph1.png")

st.markdown("""Pie chart shows the proportions of domains that does and does not contain digits. As you can see from the data that was gathered,
 a majority or ***89.1%*** to be precise of the domains from the data does not contain digits while ***10.9%*** of the domains from the data contains digits.""")

# has_punycode = df['has_punycode'].value_counts()
# colors = ['orange', 'lightgreen']
# plt.pie(has_punycode,  labels = ['URLs that does not have punycode', 'URLs that have punycode'], autopct='%1.1f%%', colors=colors)
# plt.title('Proportion of URLs with punycode')
# st.pyplot(plt)
# plt.clf()
st.image("tejadagraph2.png")

st.markdown("""Pie chart shows the proportions of URLs with and without punycode.
 From the data that was gathered it shows that only ***0.1%*** URLs have punycode while an astounding ***99.9%*** of the URLs does not have punycode.""")

# ================================================================================================================================== #

st.markdown("---")
st.markdown("### **Agor**")


st.image("agorgraph1.png")
st.markdown("""The pie chart shows the distribution of URLs with and without internal links, where the majority (***97.6%***) of URLs do not 
            contain internal links, while a smaller portion (2.4%) does contain internal links.""")


st.image("agorgraph2.png")
st.markdown("""The bar chart indicates that there are equal numbers of legitimate and phishing urls in the dataset, ***1,250,000*** for each.""")

# ================================================================================================================================== #
st.markdown("---")
st.markdown("## **Conclusion**")

st.markdown("""**Insights from our Data Visualization and Data Analysis:**
1. **Balance of the Dataset:**

    - The dataset has balanced distribution of the URLs, with each label (phishing and legitimate) having 1,250,000 samples.
    
     &nbsp;
2. **Structure and Length of URLs:**
    - Majority of the URLs in the dataset have base URLs that starts with an IP address, which is ***98.9%*** of the dataset while those that do not start with and IP address are only ***1.1%***.
    
    - According to the findings, the average length of phishing URLs in the dataset is ***71.9*** characters, while legitimate URLs have an average of ***19.9***.
    
    - Majority of the of the domains in the URLs (***89.1%***) do not contain digits, only ***10.9%*** do.
    
    - Almost all of the URLs in the dataset don't have puny codes (99.9), only ***0.1%*** of the URLs do.
    
    - A significant proportion of ***97.6%*** of the URLs do not contain internal links, only ***2.4%*** do.
    
    - These findings highlight that we can detect whether a URL is malicious based to its structure and length.""", unsafe_allow_html=True)
