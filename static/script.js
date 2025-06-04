document.addEventListener("DOMContentLoaded", function () {
    // 获取页面元素
    const uploadBtn = document.getElementById('submitBtn');
    const fileInput = document.getElementById('photo-upload');

    // 点击“上传按钮”时，触发隐藏的 file input
    uploadBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // 用户选中文件后触发上传
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) uploadPhoto(fileInput.files[0]);
    });

    // 上传照片函数
    function uploadPhoto(file) {
        if (!file.type.startsWith('image/')) {
            alert('请选择图片文件');
            return;
        }

        const formData = new FormData();
        formData.append('photo', file);

        // 向后端 /upload 发送 POST 请求
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                showResultImage(data.image_url);  // 显示识别后图片
            } else {
                alert('上传失败: ' + data.message);
            }
        })
        .catch(() => {
            alert('上传出错，请重试');
        });
    }

    function showResultImage(url) {
      const preview = document.getElementById("preview");
      const outputImage = document.getElementById("outputImage");
      const uploadSection = document.getElementById("uploadSection");

      outputImage.src = url + "?_t=" + Date.now();
      preview.classList.remove("hidden");      // 显示结果图像
      uploadSection.classList.add("hidden");   // 隐藏上传区域
    }

});
