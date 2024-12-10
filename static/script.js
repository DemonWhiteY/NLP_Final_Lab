document.getElementById("submit-btn").addEventListener("click", async () => {
    const userInput = document.getElementById("user-input").value;

    // 调用后端 API
    const response = await fetch("/process", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ input: userInput }),
    });

    const result = await response.json();

    // 显示结果
    document.getElementById("result").innerText = result.result;
});
