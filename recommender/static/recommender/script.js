// Tombol Submit
document.querySelector(".submit-btn").addEventListener("click", () => {
    alert("Upload berhasil! Tekan Ok untuk memproses.");
  });
  
  // Tombol Upload Ulang
  document.querySelector(".upload-again-btn").addEventListener("click", () => {
    document.getElementById("file-upload").value = "";
    alert("Silakan upload lagu baru.");
  });
  
  
  document.querySelector('a[href="#credits"]').addEventListener('click', function(e) {
    e.preventDefault();
    document.querySelector('#credits').scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
});

const fileInput = document.getElementById('file-upload');
const submitButton = document.getElementById('submit-btn');

// Menambahkan event listener untuk memantau perubahan di input file
fileInput.addEventListener('change', function () {
  // Jika ada file yang dipilih, aktifkan tombol submit
  if (fileInput.files.length > 0) {
    submitButton.disabled = false;
  } else {
    // Jika tidak ada file, tetap nonaktifkan tombol submit
    submitButton.disabled = true;
  }
});