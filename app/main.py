from fastapi import FastAPI, HTTPException
from app.schemas import Apartment, PredictionResponse
import joblib
import numpy as np

# Create a FastAPI application
app = FastAPI()

# Load the model

try:
    model = joblib.load("C:/Users/USER/Desktop/bina_rent_price_prediction/models/random_forest_model.joblib")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Model could not be loaded: {str(e)}")


district_target_enc={'2-ci mikrorayon q.': 400.0, '20 Yanvar m.': 771.0661290322581, '20-ci sahə q.': 667.5, '28 May m.': 1281.5741088180112, '3-cü mikrorayon q.': 619.0, '4-cü mikrorayon q.': 450.0, '5-ci mikrorayon q.': 560.0, '6-cı mikrorayon q.': 625.0, '7-ci mikrorayon q.': 732.4146341463414, '8 Noyabr m.': 1178.1826923076924, '8-ci kilometr q.': 684.4333333333333, '8-ci mikrorayon q.': 725.85, '9-cu mikrorayon q.': 562.3823529411765, 'Abşeron r.': 511.3636363636364, 'Avtovağzal m.': 577.6086956521739, 'Azadlıq Prospekti m.': 759.9142857142857, 'Ağ şəhər q.': 1833.62874251497, 'Badamdar q.': 881.1428571428571, 'Bakmil m.': 1116.25, 'Bakıxanov q.': 601.2702702702703, 'Bayıl q.': 1060.057471264368, 'Bibiheybət q.': 750.0, 'Bilgəh q.': 1100.0, 'Biləcəri q.': 542.0, 'Binə q.': 350.0, 'Binəqədi q.': 483.75, 'Binəqədi r.': 692.6530612244898, 'Buzovna q.': 528.5714285714286, 'Böyükşor q.': 1042.857142857143, 'Dərnəgül m.': 777.7215189873418, 'Elmlər Akademiyası m.': 1093.358006042296, 'Günəşli q.': 504.14285714285717, 'Gənclik m.': 966.3109404990403, 'Gəncə': 447.77777777777777, 'Hövsan q.': 463.2857142857143, 'Həzi Aslanov m.': 667.4449877750611, 'Həzi Aslanov q.': 659.3536585365854, 'Koroğlu m.': 915.5813953488372, 'Kubinka q.': 890.0, 'Köhnə Günəşli q.': 485.5, 'Lökbatan q.': 533.3333333333334, 'Lənkəran': 300.0, 'Masazır q.': 389.6421052631579, 'Massiv A q.': 450.0, 'Massiv D q.': 501.5833333333333, 'Massiv V q.': 450.0, 'Mehdiabad q.': 400.0, 'Memar Əcəmi m.': 626.4536585365854, 'Mingəçevir': 250.0, 'Məmmədli q.': 461.6666666666667, 'NZS q.': 550.0, 'Nardaran q.': 1142.15, 'Neftçilər m.': 652.0853658536586, 'Nizami m.': 1187.590425531915, 'Nizami r.': 736.5365853658536, 'Novxanı q.': 200.0, 'Nəriman Nərimanov m.': 946.3137902559868, 'Nərimanov r.': 1025.102912621359, 'Nəsimi m.': 687.25, 'Nəsimi r.': 1119.722598105548, 'Qara Qarayev m.': 761.6521739130435, 'Qaraçuxur q.': 560.7142857142857, 'Ramana q.': 410.0, 'Sabunçu q.': 430.0, 'Sabunçu r.': 847.2105263157895, 'Sahil m.': 1223.8021582733813, 'Saray q.': 396.6666666666667, 'Sea Breeze q.': 1097.0, 'Sulutəpə q.': 300.0, 'Sumqayıt': 499.78169014084506, 'Suraxanı q.': 470.0, 'Suraxanı r.': 516.125, 'Səbail r.': 1204.2962962962963, 'Ulduz m.': 666.6666666666666, 'Xalqlar Dostluğu m.': 590.8196721311475, 'Xaçmaz': 600.0, 'Xudat': 250.0, 'Xırdalan': 459.20059880239523, 'Xətai r.': 1175.798245614035, 'Xəzər r.': 1066.6666666666667, 'Yasamal q.': 736.1168831168832, 'Yasamal r.': 888.6555555555556, 'Yeni Günəşli q.': 473.68, 'Yeni Yasamal q.': 671.3647798742138, 'Zabrat q.': 446.0, 'Zığ q.': 500.0, 'İnşaatçılar m.': 727.9591194968554, 'İçəri Şəhər m.': 1133.9203539823009, 'Şah İsmayıl Xətai m.': 1210.7810760667903, 'Şamaxı': 325.0, 'Şirvan': 300.0, 'Şıxov q.': 1750.0, 'Əhmədli m.': 630.6866359447005, 'Əhmədli q.': 573.1052631578947}

district_count={'Nəriman Nərimanov m.': 1211, 'Şah İsmayıl Xətai m.': 1078, '28 May m.': 1066, 'Nizami m.': 752, 'Nəsimi r.': 739, 'Elmlər Akademiyası m.': 662, 'İnşaatçılar m.': 636, '8 Noyabr m.': 624, '20 Yanvar m.': 620, 'Gənclik m.': 521, 'Nərimanov r.': 515, 'Ağ şəhər q.': 501, 'Memar Əcəmi m.': 410, 'Həzi Aslanov m.': 409, 'Xırdalan': 334, 'Qara Qarayev m.': 322, 'Azadlıq Prospekti m.': 280, 'Sahil m.': 278, 'Neftçilər m.': 246, 'Xətai r.': 228, 'İçəri Şəhər m.': 226, 'Əhmədli m.': 217, 'Yasamal r.': 180, 'Yeni Yasamal q.': 159, 'Dərnəgül m.': 158, 'Sumqayıt': 142, 'Nəsimi m.': 132, 'Xalqlar Dostluğu m.': 122, 'Səbail r.': 108, 'Masazır q.': 95, 'Bayıl q.': 87, 'Həzi Aslanov q.': 82, 'Yasamal q.': 77, 'Badamdar q.': 77, 'Bakıxanov q.': 74, '8-ci mikrorayon q.': 60, 'Yeni Günəşli q.': 50, 'Binəqədi r.': 49, 'Sea Breeze q.': 44, 'Koroğlu m.': 43, 'Nizami r.': 41, '7-ci mikrorayon q.': 41, 'Əhmədli q.': 38, '9-cu mikrorayon q.': 34, 'Bakmil m.': 32, '8-ci kilometr q.': 30, 'Avtovağzal m.': 23, 'Hövsan q.': 21, 'Nardaran q.': 20, 'Biləcəri q.': 19, 'Sabunçu r.': 19, 'Gəncə': 18, 'Saray q.': 15, 'Massiv D q.': 12, 'Abşeron r.': 11, 'Köhnə Günəşli q.': 10, 'Binəqədi q.': 8, 'Suraxanı r.': 8, 'Buzovna q.': 7, 'Böyükşor q.': 7, 'Günəşli q.': 7, 'Qaraçuxur q.': 7, 'Lökbatan q.': 6, 'Məmmədli q.': 6, 'Zabrat q.': 5, 'Kubinka q.': 5, 'Sabunçu q.': 5, '20-ci sahə q.': 4, 'Zığ q.': 3, 'Xəzər r.': 3, 'Ulduz m.': 3, 'Şamaxı': 2, 'Şıxov q.': 2, 'Binə q.': 2, 'Ramana q.': 2, 'Şirvan': 2, 'Massiv V q.': 2, '3-cü mikrorayon q.': 2, '6-cı mikrorayon q.': 2, 'Novxanı q.': 1, 'Massiv A q.': 1, 'Lənkəran': 1, 'Xudat': 1, 'Suraxanı q.': 1, 'Mehdiabad q.': 1, 'NZS q.': 1, '5-ci mikrorayon q.': 1, 'Bibiheybət q.': 1, 'Sulutəpə q.': 1, '4-cü mikrorayon q.': 1, 'Bilgəh q.': 1, 'Mingəçevir': 1, 'Xaçmaz': 1, '2-ci mikrorayon q.': 1}
# Preprocessing function
def preprocess_data(apartment):
    # Floor ratio
    floor_ratio = apartment.floor_current / apartment.floor_total if apartment.floor_total != 0 else 0  # Avoid division by zero

    # Encode building type
    building_type_enc = 1 if apartment.building_type == 'Teze tikili' else 0

    # Encode city (Bakı / Other)
    is_baku = 1 if apartment.city == 'Bakı' else 0

    # district_freq (Count Encoding) 
    district_freq = district_count.get(apartment.district, 0)  # Default 0 if not in dictionary

    # district_target_enc (Target Encoding) 
    district_target_enc_value = district_target_enc.get(apartment.district, 0)  # Default 0 if not in dictionary

    # Return preprocessed features
    return [
        apartment.rooms,
        apartment.area_m2,
        apartment.floor_current,
        apartment.floor_total,
        floor_ratio,
        building_type_enc,
        is_baku,
        district_freq,
        district_target_enc_value
    ]

@app.post("/predict", response_model=PredictionResponse)
def predict_price(apartment: Apartment):
    try:
        # Preprocessing 
        features = preprocess_data(apartment)

        # Get predictions from the model
        predicted_price = model.predict([features])[0]

        return {"predicted_price": predicted_price}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
