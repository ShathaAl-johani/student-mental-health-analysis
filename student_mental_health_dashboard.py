import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# تحميل البيانات الأساسية
df = pd.read_csv('cleaned_student_mental_health.csv')

# تحويل القيم النصية (Yes/No) إلى 1/0 لأعمدة مثل الاكتئاب والقلق والهجمات القلقية
binary_columns = ['Do you have Depression?', 'Do you have Anxiety?', 'Do you have Panic attack?']
for col in binary_columns:
    df[col] = df[col].map({1: 1, 0: 0}).fillna(0)  # افتراض 0 إذا كانت القيمة مفقودة

# تحميل نتائج مقارنة النماذج من ملف CSV
try:
    models_results = pd.read_csv('models_comparison_results.csv')
    # إعادة تسمية الأعمدة لتتناسب مع الكود
    models_results = models_results.rename(columns={
        'Model': 'model',
        'Accuracy': 'accuracy',
        'Precision': 'precision',
        'Recall': 'recall'
    })
    # تجاهل عمود F1-Score لأنه ليس ضروريًا في الجدول الحالي
except FileNotFoundError:
    print("لم يتم العثور على ملف models_comparison_results.csv. تأكد من وجوده في نفس الدليل.")
    models_results = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall'])  # بيانات افتراضية إذا لم يتم العثور على الملف

# إنشاء تطبيق Dash
app = dash.Dash(__name__)

# تخطيط اللوحة
app.layout = html.Div([
    html.H1("تحليل الصحة النفسية للطلاب", style={'text-align': 'center', 'color': '#2c3e50'}),

    # منطقة الفلاتر (Dropdowns)
    html.Div([
        html.Label("اختر التخصص الدراسي (الكورس):"),
        dcc.Dropdown(
            id='course_filter',
            options=[{'label': i, 'value': i} for i in df['What is your course?'].unique()],
            value='Engineering',  # القيمة الافتراضية من البيانات المقدمة
            multi=True
        ),
        html.Label("اختر الفئة العمرية:"),
        dcc.Dropdown(
            id='age_filter',
            options=[
                {'label': 'كل الأعمار', 'value': 'all'},
                {'label': '<20', 'value': '<20'},
                {'label': '20-25', 'value': '20-25'},
                {'label': '>25', 'value': '>25'}
            ],
            value='all'
        ),
        html.Label("اختر الجنس:"),
        dcc.Dropdown(
            id='gender_filter',
            options=[{'label': i, 'value': i} for i in df['Choose your gender'].unique()],
            value='Female'  # القيمة الافتراضية من البيانات المقدمة
        ),
    ], style={'padding': '20px', 'background-color': '#f5f6f7', 'border-radius': '5px'}),

    # منطقة الرسوم البيانية
    html.Div([
        # مخطط 1: الإصابة بالاكتئاب حسب التخصص الدراسي (الكورس)
        dcc.Graph(id='depression_by_course'),
        
        # مخطط 2: توزيع الأعمار للمصابين وغير المصابين بالاكتئاب
        dcc.Graph(id='age_distribution'),
        
        # مخطط 3: العلاقة بين CGPA والاكتئاب
        dcc.Graph(id='cgpa_depression_relation'),
        
        # مخطط 4: الإصابة بالاكتئاب حسب الحالة الاجتماعية
        dcc.Graph(id='marital_status_depression'),
    ], style={'padding': '20px'}),

    # جدول لعرض بيانات التنبؤ
    html.Div([
        html.H3("نتائج النموذج التنبؤي", style={'text-align': 'center'}),
        dash_table.DataTable(
            id='prediction_results',
            columns=[
                {"name": "نموذج", "id": "model"},
                {"name": "الدقة (Accuracy)", "id": "accuracy"},
                {"name": "الدقة (Precision)", "id": "precision"},
                {"name": "التذكر (Recall)", "id": "recall"}
            ],
            data=models_results.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_header={'backgroundColor': '#f2f2f2', 'fontWeight': 'bold'},
            style_cell={'textAlign': 'center'}
        )
    ], style={'padding': '20px', 'background-color': '#f5f6f7', 'border-radius': '5px'})
], style={'font-family': 'Arial, sans-serif', 'margin': '20px'})

# Callback لتحديث الرسوم البيانية بناءً على الفلاتر
@app.callback(
    [Output('depression_by_course', 'figure'),
     Output('age_distribution', 'figure'),
     Output('cgpa_depression_relation', 'figure'),
     Output('marital_status_depression', 'figure')],
    [Input('course_filter', 'value'),
     Input('age_filter', 'value'),
     Input('gender_filter', 'value')]
)
def update_graphs(course, age, gender):
    # تصفية البيانات بناءً على الفلاتر
    filtered_df = df.copy()
    
    if course and not isinstance(course, str):
        filtered_df = filtered_df[filtered_df['What is your course?'].isin(course)]
    elif course:
        filtered_df = filtered_df[filtered_df['What is your course?'] == course]
    
    if gender and gender != 'all':
        filtered_df = filtered_df[filtered_df['Choose your gender'] == gender]
    
    if age == '<20':
        filtered_df = filtered_df[filtered_df['Age'] < 20]
    elif age == '20-25':
        filtered_df = filtered_df[(filtered_df['Age'] >= 20) & (filtered_df['Age'] <= 25)]
    elif age == '>25':
        filtered_df = filtered_df[filtered_df['Age'] > 25]

    # مخطط 1: الإصابة بالاكتئاب حسب التخصص الدراسي (الكورس)
    depression_by_course = filtered_df.groupby('What is your course?')['Do you have Depression?'].mean().reset_index()
    fig1 = px.bar(depression_by_course, x='What is your course?', y='Do you have Depression?',
                  title='نسبة الإصابة بالاكتئاب حسب التخصص الدراسي',
                  labels={'Do you have Depression?': 'نسبة الإصابة', 'What is your course?': 'التخصص الدراسي'})

    # مخطط 2: توزيع الأعمار للمصابين وغير المصابين بالاكتئاب
    fig2 = px.histogram(filtered_df, x='Age', color='Do you have Depression?',
                        title='توزيع الأعمار للمصابين وغير المصابين بالاكتئاب',
                        labels={'Age': 'العمر', 'Do you have Depression?': 'الإصابة بالاكتئاب'})

    # مخطط 3: العلاقة بين CGPA والاكتئاب
    # تحويل CGPA إلى قيمة رقمية إذا لم تكن كذلك
    filtered_df['What is your CGPA?'] = pd.to_numeric(filtered_df['What is your CGPA?'], errors='coerce')
    fig3 = px.scatter(filtered_df, x='What is your CGPA?', y='Do you have Depression?', color='Do you have Depression?',
                      title='علاقة المعدل التراكمي (CGPA) بالإصابة بالاكتئاب',
                      labels={'What is your CGPA?': 'المعدل التراكمي', 'Do you have Depression?': 'الإصابة بالاكتئاب'})

    # مخطط 4: الإصابة بالاكتئاب حسب الحالة الاجتماعية
    depression_by_marital = filtered_df.groupby('Marital status')['Do you have Depression?'].mean().reset_index()
    fig4 = px.pie(depression_by_marital, names='Marital status', values='Do you have Depression?',
                  title='نسبة الإصابة بالاكتئاب حسب الحالة الاجتماعية')

    return fig1, fig2, fig3, fig4

if __name__ == '__main__':
    app.run_server(debug=True)