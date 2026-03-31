async function sendData() {
  let jsonData;

  try {
    const fileInput = document.getElementById("fileInput");

    if (fileInput.files.length > 0) {
      const text = await fileInput.files[0].text();
      jsonData = JSON.parse(text);
    } else {
      jsonData = JSON.parse(document.getElementById("jsonInput").value);
    }
  } catch {
    alert("Invalid JSON");
    return;
  }

  const response = await fetch("/analyze", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(jsonData)
  });

  const result = await response.json();

  document.getElementById("output").textContent =
    JSON.stringify(result, null, 2);

  renderAllCharts(result);
}
