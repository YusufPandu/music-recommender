// Tombol Submit
document.querySelector(".submit-btn").addEventListener("click", () => {
    alert("Upload berhasil! Cari rekomendasi di bawah.");
  });
  
  // Tombol Upload Ulang
  document.querySelector(".upload-again-btn").addEventListener("click", () => {
    document.getElementById("file-upload").value = "";
    alert("Silakan upload lagu baru.");
  });
  