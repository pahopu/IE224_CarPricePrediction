import streamlit as st
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Load the pre-trained encoder and model
onehotencoder = joblib.load('deploy/data/onehotencoder.pkl')
model = joblib.load('deploy/data/best_model.pkl')

# Function to format the predicted value
def format_function(y):
    return "{:,.0f}tr".format(y / 1000000) if y < 1000000000 else "{:,.1f}tỷ".format(y / 1000000000)

# Streamlit app
def main():
    st.title("Tool dự đoán giá xe cũ")

    # Input form in two columns
    col1, col2 = st.columns(2)

    with col1:
        input_data_col1 = get_user_input_col1()
    
    with col2:
        input_data_col2 = get_user_input_col2()

    # Combine input data from both columns
    input_data = input_data_col1 + input_data_col2

    # Button to trigger processing and prediction
    if st.button("Dự đoán giá xe"):
        # Process input data
        input_df, df_onehot = process_input_data(input_data)

        # Predict using the model
        predicted = model.predict(df_onehot)

        # Format the result
        formatted_value = format_function(predicted[0])

        # Display the result in a table
        display_result(input_data, formatted_value)

def get_user_input_col1():
    year_col1 = st.number_input("Năm sản xuất", min_value=1979, max_value=2023, value=2022)
    km_driven_col1 = st.number_input("Số Km đã đi", min_value=0, value=50000)
    color_col1 = st.selectbox("Màu sắc", ['đen', 'nâu', 'đỏ', 'xanh dương', 'bạc', 'trắng', 'be', 'xanh lá', 'xám', 'tím', 'vàng', 'cam', 'ghi'])
    seller_type_col1 = st.selectbox("Phân loại người bán", ["Cá nhân", "Bán chuyên"])
    city_col1 = st.selectbox("Tỉnh", ['Hà Nội', 'Tp. Hồ Chí Minh', 'Cần Thơ', 'Tiền Giang', 'Khánh Hòa', 'Đồng Nai', 'Bình Dương', 'Hải Dương', 'Bình Phước', 'Nghệ An',
                   'An Giang', 'Lâm Đồng', 'Đà Nẵng', 'Đắk Lắk', 'Hậu Giang', 'Bắc Giang', 'Bắc Ninh', 'Bà Rịa - Vũng Tàu', 'Hải Phòng',
                   'Bình Thuận', 'Thanh Hóa', 'Vĩnh Long', 'Nam Định', 'Yên Bái', 'Thừa Thiên - Huế', 'Hà Nam', 'Sóc Trăng', 'Tây Ninh', 'Gia Lai',
                   'Hà Tĩnh', 'Bình Định', 'Đồng Tháp', 'Đắk Nông', 'Thái Bình', 'Bến Tre', 'Trà Vinh', 'Phú Yên', 'Ninh Bình', 'Kiên Giang',
                   'Quảng Nam', 'Vĩnh Phúc', 'Long An', 'Quảng Ngãi', 'Ninh Thuận', 'Phú Thọ'])
    # Add more input attributes as needed
    return [year_col1, km_driven_col1, color_col1, seller_type_col1, city_col1]

def get_user_input_col2():
    brand_col2 = st.selectbox("Hãng", ['Mitsubishi', 'Nissan', 'Ford', 'Hyundai', 'Toyota', 'Lexus',
       'Honda', 'Mercedes Benz', 'Vinfast', 'Kia', 'Audi', 'BMW',
       'Bentley', 'Mazda', 'Daihatsu', 'Peugeot', 'Chevrolet',
       'LandRover', 'Suzuki', 'MG', 'Isuzu', 'Porsche', 'Daewoo', 'Haval',
       'Acura', 'Cadillac', 'Luxgen', 'Mini', 'Infiniti', 'Volvo',
       'Ssangyong', 'Volkswagen', 'Hãng khác', 'Ferrari', 'Gaz',
       'Subaru', 'Renault', 'Jeep', 'Fiat', 'Chery', 'Jaguar', 'Skoda',
       'Mekong'])
    model_name_col2 = st.selectbox("Dòng xe", ['Xpander', 'Sunny', 'Escape', 'Accent', 'Ranger', 'Hiace', 'GX',
       'Everest', 'City', 'E Class', 'Innova', 'Fadil', 'GLC Class',
       'Fortuner', 'Elantra', 'Yaris', 'Morning', 'A6', 'Crown', '3',
       'CLA Class', '7 Series', 'Flying Spur', 'Cerato', '3 Series',
       'Corolla Altis', '6', 'Fiesta', 'Santa Fe', 'Veloz Cross',
       'Xpander Cross', 'Dòng khác', 'Lux A2.0', 'X trail', 'CR V',
       'Maybach', 'Corolla Cross', 'Grand i10', 'Charade', 'K3', 'Tucson',
       'VF8', '2008', 'Captiva', 'Veloz', 'Triton', 'Vios', 'GLC',
       'Hilux', 'Cruze', 'Spark', 'Soluto', 'Avanza', 'Range Rover Sport',
       'Zace', 'Ertiga', 'Civic', 'Sedona', 'A Class', 'Transit',
       'Attrage', 'BT 50', 'Carnival', 'Avante', 'Stargazer', 'Carens',
       'Seltos', '5', 'Grand Starex', 'Camry', 'Corolla', 'CX 5', 'Teana',
       'Hi lander', 'K5', 'Brio', 'Cayman', 'ML Class', 'X1', 'Outlander',
       'Karando', 'Sprinter', 'ZS', 'Sonata', 'Sorento', 'Laser',
       'Lancer', 'H6', 'MDX', 'Lacetti', 'Express', '5008', 'Macan',
       'Pajero Sport', '4 Series', 'Grandis', 'A7', 'Accord', 'HS',
       'Matiz', 'Astro', 'CX 8', 'Sportage', 'CTS', 'V Class',
       'Land Cruiser Prado', 'EcoSport', 'Wigo', '3008', 'Jazz', 'Kona',
       'Tiida', 'Focus', 'XL 7', 'GLE Class', 'Rondo', '5 Series',
       'Nubira', 'VFe34', 'BR-V', 'CD5', 'GLS Class', 'RX', 'Territory',
       'Raize', 'Cerato Koup', 'LS', 'S Class', 'Jolie', 'U7', 'Pajero',
       'Land Cruiser', '2', 'Swift', 'Trailblazer', 'Mulsanne', 'Cooper',
       'ES', 'Gentra', 'Mondeo', 'EX', 'Cayenne', 'RC', 'APV', 'XC60',
       'Forte', 'Stavic', 'Highlander', 'Panamera', 'Tiguan',
       'Innova Cross', 'Colorado', 'IS', 'Navara', 'CX 9', 'Sonet',
       '1083', 'Pick up', 'Chevyvan', 'VF5', 'Korando', 'Q7', 'GT Coupe',
       'Optima', 'Q8', 'Rio', 'Rush', 'Explorer', 'Range Rover Evoque',
       'VF6', 'Almera', 'Escalade', 'Grunder', 'Lux SA2.0', 'A1', 'CX 3',
       'S90', 'X6', '323', 'Magentis', 'LX', 'i20', 'GLK Class', 'Creta',
       'Prado', 'Veracruz', '718', 'Mu X', 'i30', 'Solati', 'RAV4',
       'HR-V', 'Range Rover', 'Custin', 'Gazele', 'One', 'Mirage',
       'GL Class', 'Aveo', 'Click', 'QX80', 'Vitara', 'VF9', 'Genesis',
       'Vivant', 'Forester', 'Sandero', 'M7', 'Spectra', 'Odyssey', '408',
       '4 Runner', 'CJ', 'Dmax', '911', 'Siena', 'Venza', 'Scirocco',
       'GLA Class', 'Premacy', 'X Terra', 'QQ3', 'Ciaz', 'Kyron', '86',
       'NX', 'Leganza', 'XF', 'Range Rover Vogue', 'CX-30',
       'Grand livina', 'Getz', 'Vito', 'Camaro', 'Zinger', 'GLB', 'Karoq',
       'Pride', 'Galloper', 'New Beetle', 'Q5', 'Murano', 'Yaris Cross',
       'Terracan', 'GS', 'A4', 'Kodiaq', 'Previa', 'Orlando', 'Sienna',
       'SRX', 'A5', 'Starex', 'BRZ'])
    transmission_col2 = st.selectbox("Hộp số", ['Tự động', 'Số sàn', 'Bán tự động'])
    body_type_col2 = st.selectbox("Kiểu dáng", ['Minivan (MPV)', 'Sedan', 'SUV / Cross over', 'Van', 'Hatchback',
       'Pick-up (bán tải)', 'Kiểu dáng khác', 'Coupe (2 cửa)', 'Mui trần'])
    origin_col2 = st.selectbox("Xuất xứ", ['Đang cập nhật', 'Hàn Quốc', 'Việt Nam', 'Nước khác', 'Nhật Bản',
       'Thái Lan', 'Đức', 'Đài Loan', 'Mỹ', 'Ấn Độ', 'Trung Quốc'])
    # Add more input attributes as needed
    return [brand_col2, model_name_col2, transmission_col2, body_type_col2, origin_col2]

def process_input_data(input_data):
    column_names = ["Năm sản xuất", "Số Km đã đi", "Màu sắc", "Phân loại người bán", "Tỉnh", "Hãng", "Dòng xe", "Hộp số", "Kiểu dáng", "Xuất xứ"]
    input_df = pd.DataFrame([input_data], columns=column_names)
    categorical_columns = input_df.select_dtypes(include=['object']).columns
    X_categorical = input_df[categorical_columns]
    X_encoded = onehotencoder.transform(X_categorical)
    df_onehot = pd.concat([input_df.drop(columns=categorical_columns), pd.DataFrame(X_encoded.toarray(), columns=onehotencoder.get_feature_names_out(categorical_columns))], axis=1)
    return input_df, df_onehot

def display_result(input_data, formatted_value):
    st.subheader("Bảng dự đoán")
    result_df = pd.DataFrame({"Các thông số của xe": ["Năm sản xuất", "Số Km đã đi", "Màu sắc", "Phân loại người bán", "Tỉnh", "Hãng", "Dòng xe", "Hộp số", "Kiểu dáng", "Xuất xứ"],
                              "Giá trị": input_data})
    result_df = result_df.append({"Các thông số của xe": "Giá dự đoán", "Giá trị": formatted_value}, ignore_index=True)
    st.table(result_df)

if __name__ == "__main__":
    main()
