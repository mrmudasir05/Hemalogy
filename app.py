import dash
from dash import dcc  # dash-core-components now integrated directly into dash
from dash import html  # dash-html-components now integrated directly into dash
from dash import dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import base64
import tempfile
import io
import cv2
from main_calls import main
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import plotly.graph_objs as go  # plotly.graph_objs is fine


# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': '0',
    'left': '0',
    'bottom': '0',
    'width': '200px',  # Slightly wider for better spacing
    'padding': '12px',
    'backgroundColor': '#124f8f',  # Dark background color
    'color': 'white',  # White text color
    'fontFamily': 'Arial, sans-serif',  # Clean font family
    'boxShadow': '2px 0 5px rgba(0,0,0,0.1)',  # Subtle shadow for depth
    'overflowY': 'auto'  # Allow vertical scrolling if needed
}

# Define link style for the sidebar links
LINK_STYLE = {
    'color': '#eeefef',
    'textDecoration': 'none',  # Remove underline
    'padding': '10px 10px',
    'display': 'block',
    'borderRadius': '4px',
    'transition': 'background-color 0.3s',
}

# Define hover effect for the links
LINK_HOVER_STYLE = {
    'backgroundColor': '#495057'  # Darker background on hover
}

CONTENT_STYLE = {
    'margin-left': '110px',  # Make space for the sidebar
    'padding': '10px',
    'background-color': '#add8e6',  # Light blue background color
    'font-family': 'Arial, sans-serif',  # Font style
    'color': '#333',  # Text color
    'line-height': '1.6'  # Line height for better readability
}

# Helper function to process image
def process_image(image_path):
    processed_image, data = main(image_path)
    # Convert data to DataFrame
    df = pd.DataFrame(list(data.items()), columns=['Cell Type', 'Count'])
    return processed_image, df

# Define the layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # Location component to manage URL
    html.Div([
        # Sidebar
        html.Div([
            html.Div([
                html.H2('Navigation', style={'margin-bottom': '40px', 'background-color': '#ad8e6', 'color': 'white', 'width': '100%', 'text-align': 'center'})
            ]),
            html.Div([
                html.Hr(),
                dcc.Link('About', href='/', style=LINK_STYLE),  # Link to about page
                html.Hr(),
                dcc.Link('Classifier', href='/classifier', style=LINK_STYLE),  # Link to classifier page
                html.Hr(),
            ])
        ], style=SIDEBAR_STYLE),

        # Main content
        html.Div(id='page-content', style=CONTENT_STYLE)  # Placeholder for page content
    ])
])

# Callback to render page content based on URL pathname
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def render_page_content(pathname):
    if pathname == '/':
        return html.Div([
            html.Header([
                html.Div([
                    html.Img(src='/assets/iub.jpg', style={
                        'height': '80px', 'width': '80px', 'border-radius': '50%', 'float': 'left', 'margin-right': '10px'
                    }),
                    html.H1("Hematology Analyzer", style={
                        'textAlign': 'center', 'display': 'inline-block', 'width': '80%', 'color': 'white'
                    }),
                    html.Img(src='/assets/deepembed.jpg', style={
                        'height': '85px', 'width': '90px', 'border-radius': '50%', 'float': 'right', 'margin-left': '10px'
                    }),
                ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between'}),
                html.H2("Solution of problems", style={'textAlign': 'center'}),
            ], style={'backgroundColor': '#124f8f', 'padding': '18px'}),
            html.Div([
                html.H2("About this application"),
                html.P("Blood tests are a crucial diagnostic tool in healthcare, enabling the detection of various diseases by analyzing different types of red blood cells (RBCs). Abnormalities in specific RBC types can indicate specific blood disorders, allowing for targeted diagnoses."),
                html.P("Distinctive RBC shapes can be indicative of particular health issues; for instance, tear drop cells are associated with extramedullary hemopoiesis, while an overabundance of echinocytes can signal severe liver disease. Traditionally, manual methods and hematology analyzers have been employed to identify RBC deformities."),
                html.P("The advent of Artificial Intelligence (AI) is transforming the medical landscape, offering enhanced efficiency and accuracy in diagnosing diseases. By leveraging AI, healthcare professionals can obtain more precise results in significantly less time, facilitating earlier disease detection and treatment."),
            ], style={'padding': '20px'}),
            # Footer
            html.Footer([
        html.Div([
            # Contact Information
            html.Div([
                html.H3("Contact Us", style={'color': 'white', 'margin-bottom': '5px'}),
                html.P("Email: mailtousman8@gmail.com", style={'color': 'white', 'margin': '0', 'line-height': '1.2'}),
                html.Br(),
                html.P("Email: mrmudasir05@gmail.com", style={'color': 'white', 'margin': '0', 'line-height': '1.2'}),
            ], className="footer-section"),

            # Social Media Links
            html.Div([
                html.H3("Follow Us", style={'color': 'white', 'margin-bottom': '20px', 'text-align': 'center'}),
                html.A(html.Img(src="/assets/github-icon.png", className="social-icon", style={'height': '30px', 'width': '30px', 'margin-right': '20px', 'border-radius': '50%'}), href="https://github.com/mrmudasir05", target="_blank"),
                html.A(html.Img(src="/assets/linkedin-icon.png", className="social-icon", style={'height': '30px', 'width': '30px', 'margin-right': '20px', 'border-radius': '50%'}), href="https://www.linkedin.com/in/mudasir-azhar-a80b68237/", target="_blank"),
                html.A(html.Img(src="/assets/linkedin-icon.png", className="social-icon", style={'height': '30px', 'width': '30px', 'margin-right': '20px', 'border-radius': '50%'}), href="https://www.linkedin.com/in/musman8?trk=contact-info", target="_blank"),
                html.P("© 2024 DeepEmbed. All rights reserved.", style={'color': 'white', 'margin-bottom': '0', 'line-height': '1.2', 'text-align': 'center'})
            ], className="footer-section", style={'text-align': 'center', 'padding': '0px'}),

            # Map
            
        ], className="footer-container", style={'display': 'flex', 'justify-content': 'space-between', 'flex-wrap': 'wrap'})

    ], style={'backgroundColor': '#124f8f', 'padding': '20px', 'max-width': '1000px', 'margin': '20px'})


        ], style=CONTENT_STYLE)

    elif pathname == '/classifier':
        return html.Div([
            html.Header([
                html.Div([
                    html.Img(src='/assets/iub.jpg', style={
                        'height': '80px', 'width': '80px', 'border-radius': '50%', 'float': 'left', 'margin-right': '10px'
                    }),
                    html.H1("Hematology Analyzer", style={
                        'textAlign': 'center', 'display': 'inline-block', 'width': '80%', 'color': 'white'
                    }),
                    html.Img(src='/assets/deepembed.jpg', style={
                        'height': '85px', 'width': '90px', 'border-radius': '50%', 'float': 'right', 'margin-left': '10px'
                    }),
                ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between'}),
                html.H2("Solution of problems", style={'textAlign': 'center'}),
            ], style={'backgroundColor': '#124f8f', 'padding': '13px'}),

            html.P("This dashboard shows the original and processed images along with their respective metrics."),
            html.Div([
                dcc.Input(id='input-1', type='text', placeholder='Animal tag', style={
                    'margin': '15px',
                    'width': '200px',   # Adjust width as needed
                    'height': '30px',   # Adjust height as needed
                    'font-size': '13px' # Adjust font size as needed
                }),
                dcc.Input(id='input-2', type='number', placeholder='Picture tag', style={
                    'margin': '15px',
                    'width': '200px',   # Adjust width as needed
                    'height': '30px',   # Adjust height as needed
                    'font-size': '13px' # Adjust font size as needed
                }),
            ], style={'margin-bottom': '40px'}),

            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select an Image')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
            ),

            html.Div([
                html.Div([
                    html.Img(id='original-image', style={'width': '100%'}),
                    html.H3("Original Image")
                ], style={'width': '45%', 'display': 'inline-block', 'padding': '10px'}),

                html.Div([
                    html.Img(id='processed-image', style={'width': '100%'}),
                    html.H3("Processed Image")
                ], style={'width': '45%', 'display': 'inline-block', 'padding': '10px'})
            ], style={'text-align': 'center'}),

            dash_table.DataTable(
                id='image-report-table',
                columns=[{"name": "Cell Type", "id": "Cell Type"}, {"name": "Count", "id": "Count"}],
                style_table={'width': '70%', 'margin': 'auto', 'margin-top': '50px'},
                style_cell={'textAlign': 'center'},
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                }
            ),

            html.Div([
                html.Button("Download Report", id="download-button"),
                dcc.Download(id="download-report")
            ], style={'text-align': 'center', 'margin': '20px'}),
            html.Footer([
                html.Div([
            # Contact Information
            html.Div([
                html.H3("Contact Us", style={'color': 'white', 'margin-bottom': '5px'}),
                html.P("Email: mailtousman8@gmail.com", style={'color': 'white', 'margin': '0', 'line-height': '1.2'}),
                html.Br(),
                html.P("Email: mrmudasir05@gmail.com", style={'color': 'white', 'margin': '0', 'line-height': '1.2'}),
            ], className="footer-section"),

            # Social Media Links
            html.Div([
                html.H3("Follow Us", style={'color': 'white', 'margin-bottom': '20px', 'text-align': 'center'}),
                html.A(html.Img(src="/assets/github-icon.png", className="social-icon", style={'height': '30px', 'width': '30px', 'margin-right': '20px', 'border-radius': '50%'}), href="https://github.com/mrmudasir05", target="_blank"),
                html.A(html.Img(src="/assets/linkedin-icon.png", className="social-icon", style={'height': '30px', 'width': '30px', 'margin-right': '20px', 'border-radius': '50%'}), href="https://www.linkedin.com/in/mudasir-azhar-a80b68237/", target="_blank"),
                html.A(html.Img(src="/assets/linkedin-icon.png", className="social-icon", style={'height': '30px', 'width': '30px', 'margin-right': '20px', 'border-radius': '50%'}), href="https://www.linkedin.com/in/musman8?trk=contact-info", target="_blank"),
                html.P("© 2024 DeepEmbed. All rights reserved.", style={'color': 'white', 'margin-bottom': '0', 'line-height': '1.2', 'text-align': 'center'})
            ], className="footer-section", style={'text-align': 'center', 'padding': '0px'}),

            # Map
            
        ], className="footer-container", style={'display': 'flex', 'justify-content': 'space-between', 'flex-wrap': 'wrap'})

            ], style={'backgroundColor': '#124f8f', 'padding': '13px', 'max-width': '1000px', 'margin': 'auto'})

        ], style=CONTENT_STYLE)
        
    

    # elif pathname == '/classifier':
    #     return html.Div([
    #         html.Header([
    #             html.H1("Classifier Page", style={'textAlign': 'center'})
    #         ], style={'backgroundColor': '#124f8f', 'padding': '13px'}),
    #         html.Div([
    #             html.H2("Classifier information"),
    #             html.P("Information about the image classification model used in this application."),
    #             html.P("Developed using machine learning techniques."),
    #         ], style={'padding': '20px'})
    #     ], style=CONTENT_STYLE)

    else:
        return html.Div([
            html.Header([
                html.H1("Page not found", style={'textAlign': 'center'})
            ], style={'backgroundColor': '#124f8f', 'padding': '13px'}),
            html.Div([
                html.H2("404 Error"),
                html.P(f"The requested URL {pathname} was not found on this server."),
            ], style={'padding': '20px'})
        ], style=CONTENT_STYLE)


# Callback to handle image upload and processing
@app.callback(
    [Output('original-image', 'src'),
     Output('processed-image', 'src'),
     Output('image-report-table', 'data')],
    [Input('upload-image', 'contents')],
    [State('upload-image', 'filename')]
)
def update_images(contents, filename):
    if contents is not None:
        # Decode the image
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        # Save the uploaded image temporarily using tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(decoded)
            temp_filename = temp_file.name  # Get the temporary file name

        # Process the image using the main function
        processed_image, df = process_image(temp_filename)

        # Convert the processed image to base64
        _, buffer = cv2.imencode('.png', processed_image)
        processed_image_base64 = base64.b64encode(buffer).decode()

        # Convert DataFrame to list of dictionaries
        table_data = df.to_dict('records')

        # Convert images to base64
        original_image_src = f'data:image/png;base64,{base64.b64encode(decoded).decode()}'
        processed_image_src = f'data:image/png;base64,{processed_image_base64}'

        # Optionally, clean up the temp file after processing if delete=True
        return original_image_src, processed_image_src, table_data
    else:
        return None, None, []

# Callback to download the report
@app.callback(
    Output("download-report", "data"),
    Input("download-button", "n_clicks"),
    State('upload-image', 'contents'),
    State('upload-image', 'filename'),
    State('image-report-table', 'data'),
    State('input-1', 'value'),
    State('input-2', 'value'),
    State('url', 'pathname'),
    prevent_initial_call=True
)
def generate_report(n_clicks, contents, filename, table_data, input1_value, input2_value, pathname):
    if contents and table_data:
        # Decode the image
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Create a PDF
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        
        # Add input values to PDF
        pdf.drawString(50, 300, f"Animal Tag: {input1_value}")
        pdf.drawString(50, 280, f"Picture Tag: {input2_value}")

        # Add original image to PDF
        original_image = ImageReader(io.BytesIO(decoded))
        pdf.drawImage(original_image, 50, 500, width=200, height=200)
        
        # Process the image to get processed image
        processed_image, _ = process_image(filename)
        _, buffer_processed = cv2.imencode('.png', processed_image)
        processed_image_base64 = base64.b64encode(buffer_processed).decode()
        processed_image_reader = ImageReader(io.BytesIO(base64.b64decode(processed_image_base64)))
        
        # Add processed image to PDF
        pdf.drawImage(processed_image_reader, 300, 500, width=200, height=200)
        
        # Add table data to PDF
        pdf.drawString(50, 450, "Cell Type Counts")
        table_start_y = 430
        for row in table_data:
            pdf.drawString(50, table_start_y, f"{row['Cell Type']}: {row['Count']}")
            table_start_y -= 20
        
        pdf.save()
        
        # Prepare PDF for download
        buffer.seek(0)
        pdf_data = buffer.getvalue()
        
        return dcc.send_bytes(pdf_data, f"{pathname}_report.pdf")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
