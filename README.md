# Laporan UAS Praktikum AI

**Anggota Kelompok :**

1. Ade Prasetyo 		 			3.34.21.3.01
2. Ferina Ayu Fella Puspita   3.34.21.3.11

## Domain Proyek

*Mobil* di era digital saat ini menjadi barang yang sangat esensial untuk mobilitas dan transportasi. Menurut data dari Asosiasi Penyelenggara Jasa Internet Indonesia (APJII) tahun 2022, sebagian besar penduduk Indonesia memiliki mobil sebagai alat transportasi. Harga *mobil* menjadi parameter yang paling berpengaruh terhadap kecenderungan seseorang untuk membeli mobil. Oleh karena itu, kemampuan *machine learning* dalam memprediksi harga mobil menjadi sangat penting bagi masyarakat yang ingin mencari mobil sesuai kebutuhan mereka.
Dalam proyek "Car Price Prediction" (Prediksi Harga Mobil), terdapat beberapa parameter yang dapat digunakan untuk memprediksi harga mobil. Beberapa parameter tersebut meliputi tahun produksi, merek mobil, model mobil, jarak tempuh, kondisi mobil, jenis bahan bakar, kapasitas mesin, transmisi, jumlah pemilik sebelumnya, dan sebagainya. 

Salah satu bidang penelitian machine learning yang dapat digunakan untuk melakukan prediksi harga mobil adalah regresi. Algoritma regresi dapat memprediksi nilai dari variabel target, yaitu harga mobil, berdasarkan nilai dari beberapa variabel input atau fitur seperti yang disebutkan di atas.

Ada beberapa penelitian yang telah dilakukan terkait prediksi harga mobil menggunakan machine learning. Misalnya, penelitian yang dilakukan oleh [3] menggunakan beberapa algoritma machine learning seperti Random Forest, Logistic Regression, Decision Tree, Linear Discriminant Analysis, K-Nearest Neighbor, dan SVC. Hasil penelitian menunjukkan bahwa algoritma SVC memiliki akurasi yang paling baik dalam memprediksi harga mobil.

Selain itu, penelitian yang dilakukan oleh [4] juga melakukan prediksi harga mobil menggunakan 25 jenis algoritma machine learning. Hasil pengujian menunjukkan bahwa algoritma SVM memiliki akurasi tertinggi, yaitu 0.9470.

Dalam proyek ini, Anda dapat melakukan prediksi harga mobil menggunakan beberapa algoritma machine learning yang telah terbukti efektif seperti Random Forest, Logistic Regression, Decision Tree, Linear Discriminant Analysis, K-Nearest Neighbor, SVC, atau SVM. Sebelum melatih model dan melakukan evaluasi, penting untuk melakukan manipulasi data dan pembersihan data agar model machine learning yang dihasilkan memiliki akurasi yang tinggi.

## Business Understanding

Setiap tahun terdapat berbagai jenis tipe mobil dengan fitur yang bermacam-macam. Umumnya, jenis mobil tertentu mengalami peningkatan perangkatnya, seperti peningkatan ukuran mesin, jumlah tenaga kuda, fitur keamanan, dan teknologi yang tertanam. Penambahan fitur pada mobil akan mempengaruhi kenaikan harga mobil tersebut. Oleh karena itu, calon pembeli harus memahami spesifikasi mobil yang akan dibeli sesuai dengan anggaran yang dimiliki. Diperlukan sebuah model machine learning yang dapat memprediksi harga mobil dengan akurat. Sehingga calon pembeli dapat menyiapkan anggaran yang digunakan untuk membeli mobil dengan spesifikasi yang diinginkan.

#### Problem Statements

1. Berdasarkan permasalahan yang telah dijelaskan, *problem statements* dari proyek "Car Price Prediction" adalah sebagai berikut:

   1. Fitur-fitur apa saja yang paling berpengaruh terhadap harga mobil?
   2. Bagaimana proses *Pre-Processing* yang dilakukan agar menghasilkan model *machine learning* yang akurat dalam memprediksi harga mobil?
   3. Bagaimana membuat atau memilih model *machine learning* yang memiliki akurasi terbaik dalam memprediksi harga mobil?

#### Goals

1. Mengetahui fitur yang paling berkorelasi dengan penentuan harga mobil:
   Tujuan ini melibatkan analisis statistik dan eksplorasi data untuk mengidentifikasi fitur-fitur mobil yang memiliki hubungan yang kuat dengan harga jualnya. Anda dapat menggunakan metode seperti korelasi, analisis regresi, atau teknik visualisasi data untuk memahami hubungan ini.
2. Membuat model machine learning yang dapat memprediksi harga mobil:
   Tujuan ini melibatkan pembangunan model machine learning yang dapat mempelajari pola dari data mobil yang ada dan melakukan prediksi harga mobil berdasarkan fitur-fiturnya. Anda dapat menggunakan berbagai algoritma machine learning seperti regresi linier, regresi non-linier, atau algoritma pembelajaran mesin lainnya untuk membangun model prediksi.

3. Memilih model machine learning yang menghasilkan prediksi paling akurat berdasarkan proses preprocessing yang dilakukan:
   Tujuan ini melibatkan evaluasi dan pemilihan model machine learning yang menghasilkan prediksi harga mobil paling akurat. Proses preprocessing data juga menjadi penting dalam meningkatkan kualitas prediksi. Anda perlu melakukan pemrosesan data seperti pengisian nilai yang hilang, normalisasi fitur, atau pemilihan fitur yang relevan untuk meningkatkan kualitas prediksi. Setelah itu, Anda dapat membandingkan kinerja berbagai model menggunakan metrik evaluasi seperti Mean Absolute Error (MAE), Mean Squared Error (MSE), atau Coefficient of Determination (R^2) untuk memilih model yang paling akurat.

####  Solution Statements

1. Melakukan analisis deskriptif untuk mengetahui pola dan informasi yang tersimpan di data mengenai fitur atau spesifikasi yang mempengaruhi harga mobil.
2. Melakukan proses Data Manipulation untuk menggabungkan kolom-kolom yang berpengaruh terhadap akurasi prediksi harga mobil.
3. Melakukan Proses Preprocessing seperti:

- Mengecek missing value dan duplikasi data. Kebetulan dataset yang digunakan tidak ada missing value dan duplikasi data.
- Menghapus kolom yang tidak berpengaruh terhadap prediksi harga.
- Melakukan visualisasi data untuk melihat persebaran dan korelasi antar kolom.
- Membagi data menjadi training dan test set, dengan prosentase 85% banding 15%. Alasan menggunakan 15% karena jumlah data yang digunakan banyak, jadi hanya dengan 15% sudah didapatkan banyak data tes.
- Melakukan Encoding terhadap kolom yang bertipe objek/kategorikal menggunakan fungsi Map.

## Data Understanding

Dataset yang digunakan pada proyek ini diperoleh dari Kaggle. Silahkan kunjungi tautan berikut [Car Price Prediction](https://www.kaggle.com/datasets/sidharth178/car-prices-dataset?datasetId=1479517) untuk mengakses dataset yang dipakai. Adapun variabel-variabel yang terdapat pada dataset adalah sebagai berikut :

1.car_name: Nama mobil.
2.year: Tahun pembuatan mobil.
3.selling_price: Harga penjualan mobil dalam mata uang tertentu.
4.km_driven: Jarak tempuh mobil dalam kilometer.
5.fuel_type: Jenis bahan bakar mobil (misalnya, Diesel, Petrol, CNG).
6.seller_type: Tipe penjual (misalnya, Dealer, Individual).
7.transmission: Jenis transmisi mobil (misalnya, Manual, Otomatis).
8.owner: Jumlah pemilik sebelumnya yang dimiliki mobil.
9.mileage: Jarak yang ditempuh mobil per liter bahan bakar.
10.engine: Kapasitas mesin mobil dalam cc (centimeter kubik).
11.max_power: Daya maksimum yang dihasilkan oleh mesin dalam bhp (tenaga kuda brake).
12.seats: Jumlah kursi yang tersedia di mobil.

Pada dataset, terdapat 12 fitur numerikal dan 9 fitur kategorikal. Ringkasan statistik dari data-data numerikal dapat dilihat pada Tabel 1.

<div style="text-align:center">Tabel 1. Ringkasan Statistik Data-Data Numerikal Pada Dataset</div>

![image-20230712120204238](C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20230712120204238.png)

![image-20230712120312617](C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20230712120312617.png)

Pada tahap *Data Understanding* dilakukan analisis data eksploratif untuk mendapatkan wawasan tentang karakteristik data, memahami struktur data, dan mengidentifikasi potensi masalah atau kesalahan yang mungkin terjadi. Kegiatan Data Understanding yang dilakukan pada Proyek ini antara lain :

1.Informasi Dasar:
Memberikan informasi tentang jumlah data yang tersedia dalam dataset.
Mengidentifikasi dan menangani nilai yang hilang (missing value) jika ada.
Memeriksa keberadaan duplikasi data dan menghapusnya jika diperlukan.

2.Manipulasi Data:
Melakukan manipulasi data untuk mendapatkan variabel atau fitur baru yang dapat berguna dalam analisis, seperti menggabungkan kolom atau menghitung metrik baru.
Misalnya, dalam konteks "Car Price Prediction", Anda dapat melakukan manipulasi data untuk mendapatkan variabel baru seperti usia mobil (berdasarkan tahun pembuatan) atau konversi kapasitas mesin dari liter ke cc.

3.Korelasi Antar Kolom:
Menganalisis korelasi antar kolom untuk mengidentifikasi hubungan atau asosiasi antara variabel-variabel dalam dataset.
Anda dapat menggunakan metode seperti korelasi Pearson untuk mengukur korelasi linier antara variabel numerikal.
Misalnya, dalam proyek ini, Anda dapat mengidentifikasi variabel yang berkorelasi dengan harga mobil, seperti tahun pembuatan, jarak tempuh, atau kapasitas mesin.

4.Visualisasi Data:
Melakukan visualisasi data untuk memahami sebaran dan hubungan antara variabel-variabel.
Anda dapat menggunakan plot seperti histogram, scatter plot, atau box plot untuk melihat distribusi dan sebaran data.
Visualisasi korelasi antar kolom dapat dilakukan dengan menggunakan heatmap atau matriks korelasi.
Dalam konteks proyek "Car Price Prediction", Anda dapat menghasilkan visualisasi yang menunjukkan korelasi antara variabel numerikal, seperti harga mobil dengan kapasitas mesin, usia mobil, atau jarak tempuh.



![image-20230712162140201](C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20230712162140201.png)

<div style="text-align:center">Gambar 1. Visualisasi Korelasi Data dengan Heatmap</div>

Sedangkan contoh visualisasi dari sebaran data ditunjukan pada Gambar 2. Visualisasi yang ditunjukan pada Gambar 2 menunjukan bahwa ada ketidakseimbangan data pada kolom Battery, Screen_Size, Processor, dan RAM.

![image-20230712162252102](C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20230712162252102.png)

<div style="text-align:center">Gambar 2. Visualisasi Data pada Dataset</div>

## Data Preparation

Teknik data preparation yang dilakukan pada proyek ini adalah sebagai berikut : 

1. Melakukan seleksi fitur yaitu membagi data fitur dan data label

2. Membagi dataset menjadi data training dan data testing

   **Membagi dataset menjadi data latih dan data uji dengan prosentasi 80 : 20**

   ![image-20230712162508840](C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20230712162508840.png)

![image-20230712162542841](C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20230712162542841.png)

## Modelling

Pada tahap ini dilakukan proses pelatihan untuk mendapatkan model dengan performa terbaik. Tahapan yang dilakukan pada proses Modelling adalah sebagai berikut.

Lazy Predict adalah library Python yang membantu dalam membuat model *machine learning* dengan cepat dan mudah. Library lazypredict gagal di import sehingga failed saat di didownload karena ada kesalahan pada dataset.

<div style="text-align:center">Tabel 2. Hasil Perbandingan model menggunakan Lazy Predict</div>

Mencari hyperparameter GradienBossting

![image-20230712195759510](C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20230712195759510.png)

Mencari hyperparameter RandomForest

![image-20230712195907737](C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20230712195907737.png)

Mencari hyperparameter BaggingRegressor

![image-20230712195956915](C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20230712195956915.png)

## Evaluation

- Metrik evaluasi yang digunakan adalah *Mean Square Error* (MSE), *Root Mean Square Error* (RMSE), dan *R Square* (R2 Score)

- MSE melakukan pengurangan nilai data aktual dengan data peramalan dan hasilnya dikuadratkan (*squared*) kemudian dijumlahkan secara keseluruhan dan membaginya dengan banyaknya data yang ada. Rumus dari MSE adalah sebagai berikut 
  $$
  MSE = \frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2
  $$
  Diketahui:

  - n = Jumlah Data
  - yi = *Actual Value* / Nilai Sebenarnya
  - ŷi = *Predicted Value* / Nilai Prediksi

- RMSE adalah jumlah dari kesalahan kuadrat atau selisih antara nilai sebenarnya dengan nilai prediksi yang telah ditentukan. Cara menghitungnya tinggal mengakar kan mse menggunakan fungsi *np.sqrt*. Rumus dari RMSE adalah sebagai berikut.
  $$
  RMSE = \sqrt{(\frac{1}{n})\sum_{i=1}^{n}(y_{i} - x_{i})^{2}}
  $$
  Diketahui:

  - n = Jumlah Data
  - yi = *Actual Value* / Nilai Sebenarnya
  - ŷi = *Predicted Value* / Nilai Prediksi

- R2 Score dijadikan sebagai pengukuran seberapa baik garis regresi mendekati nilai data asli yang dibuat melalui model. Rumus dari R2 Score adalah sebagai berikut.
  $$
  R^2 = 1 - {SS_R \over SS_T} =  1 - {\sum_{i} (y_i - ŷ_p) ^ 2 \over \sum_{i} (y_i - ȳ) ^ 2}
  $$

  - SSR : Kuadrat dari selisih nilai Y prediksi dengan nilai rata-rata Y = ∑ (Ypred – Yrata-rata)²
  - SST : Kuadrat dari selisih nilai Y aktual dengan nilai rata-rata Y = ∑ (Yaktual – Yrata-rata)²

- Setelah melalui tahap pelatihan dan evaluasi menggunakan MSE, RMSE, dan R Square, diperoleh hasil bahwa algoritma **GradienBoosterRegressor** memiliki performa yang paling baik seperti ditunjukan Tabel 3. Maksud performa yang paling baik adalah memiliki nilai MSE dan RMSE yang mendekati nilai 0, serta memiliki nilai R2 Score yang mendekati nilai 1.

  <div style="text-align:center">Tabel 3. Hasil Pengujian dari 3 Algoritma Teratas</div>

  ![image-20230712201252828](C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20230712201252828.png)

- Membandingkan data sebenarnya dengan hasil prediksi. Hasil perbandingan dapat dilihat pada Tabel 4.

  <div style="text-align:center">Tabel 4. Hasil Perbandingan Data Sebenarnya dengan Hasil Prediksi</div>

![image-20230712201326115](C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20230712201326115.png)

## Conclussion

1. Berdasarkan hasil pengukuran, terdapat 10 kolom atau fitur yang mempengaruhi *Price* yaitu Manufacturer, Model, Prod.Year, Category, Fuel type, Engine Volume, Mileage, Gear box type, Drive wheels, Doors, Wheel, Color, dan Airbags.  
2. Proses preprocessing yang dilakukan adalah dengan melakukan manipulasi data seperti mengabungkan Resolution X dan Resolution Y untuk menghasilkan fitur baru yaitu PPI. Menghapus data yang tidak memiliki korelasi yang signifikan dengan *Price*, dan mengubah format tipe data pada setiap kolom yang memiliki korelasi.
3. Berdasarkan hasil pengujian model, diperoleh hasil bahwa algoritma GradienBoosting memiliki performa yang paling baik yaitu memiliki nilai RMSE paling kecil dan R2 Score paling besar.
4. Meningkatkan performa model dapat dilakukan dengan menambahkan hyperparameter.  Pemilihan hyperparameter yang menghasilkan performa terbaik dapat dilakukan menggunakan teknik Grid Search.
5. Dataset yang digunakan memiliki rentang jangkauan yang berbeda (imbalace), oleh sebab itu agar performa model lebih baik maka perlu dilakukan teknik SMOTE untuk menangani imbalance dataset.

## Referensi

[1]   https://repository.its.ac.id/44824/1/5115201015-Master_Thesis.pdf

[2]   https://core.ac.uk/download/pdf/235044836.pdf

[3]   https://www.dataindustri.com/produk/data-report-tren-data-penjualan-mobil-berdasarkan-merek-dan-tipe-di-indonesia-januari-desember-2020/

[4]  http://e-journal.uajy.ac.id/27764/1/18%2007%2009861%200.pdf

[5]  http://eprints.uniska-bjm.ac.id/5404/1/ARTIKEL%20RISKY%20MAULANA.pdf

[6]  https://www.dataindustri.com/produk/laporan-data-tren-data-penjualan-mobil-berdasarkan-merek-sub-merek-dan-tipe-di-indonesia-2021/



**---Ini adalah bagian akhir laporan---**
