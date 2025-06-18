from django.shortcuts import render, redirect
from django.urls import reverse
import joblib
import pandas as pd
import shap
import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

model = joblib.load("rent_model.pkl")
explainer = shap.TreeExplainer(model)
plt.ioff()

def tahmin_view(request):
    context = {
        "tahmin": None,
        "shap_img": None,
        "hata": None,
    }

    if request.method == "POST":
        try:
            metrekare = float(request.POST.get("metrekare"))
            oda_sayisi = float(request.POST.get("oda_sayisi"))
            bina_yasi = int(request.POST.get("bina_yasi"))
            aidat = int(request.POST.get("aidat"))
            isitma = int(request.POST.get("isitma"))
            kat_grup = int(request.POST.get("kat"))
            mobilya = int(request.POST.get("mobilya"))

            columns = ['Net m²', 'Oda Sayısı', 'Bina Yaşı', 'Isıtma Tipi',
                       'Kat Grup', 'Aidat', 'Mobilya Durumu']

            veri = pd.DataFrame([[metrekare, oda_sayisi, bina_yasi, isitma, kat_grup, aidat, mobilya]],
                                columns=columns)
            veri = veri.round(0)
            tahmin = round(model.predict(veri)[0])

            shap_values = explainer.shap_values(veri)
            shap_array = np.array(shap_values).flatten()
            sorted_idx = np.argsort(np.abs(shap_array))[::-1][:7]
            sorted_features = [columns[i] for i in sorted_idx]
            sorted_values = shap_array[sorted_idx]

            fig, ax = plt.subplots(figsize=(8, len(sorted_features) * 0.4))
            y_pos = np.arange(len(sorted_features))
            colors = ["red" if v > 0 else "blue" for v in sorted_values]
            ax.barh(y_pos, sorted_values, color=colors, height=0.3)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_features, fontsize=12, ha='right')
            ax.set_xlabel("Etki", fontsize=12)
            ax.set_title("Özellik Katkısı", fontsize=14)
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            shap_img = f"data:image/png;base64,{img_base64}"

            # ✅ Session'a kaydet
            request.session['tahmin'] = f"{tahmin:,} TL".replace(",", ".")
            request.session['shap_img'] = shap_img

            # ✅ Redirect (PRG pattern)
            return redirect(reverse("tahmin_view"))

        except Exception as e:
            request.session['hata'] = f"Hata: {str(e)}"
            return redirect(reverse("tahmin_view"))

    else:
        # ✅ GET ise session'dan çek ve hemen sil
        context["tahmin"] = request.session.pop("tahmin", None)
        context["shap_img"] = request.session.pop("shap_img", None)
        context["hata"] = request.session.pop("hata", None)

    return render(request, "tahmin/form.html", context)
