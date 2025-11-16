using System.Diagnostics;
using System.Text.Json;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();

app.MapPost("/api/solve", async (SolveRequest req, HttpContext ctx) =>
{
    if (string.IsNullOrWhiteSpace(req.Facelets) || req.Facelets.Length != 54)
        return Results.BadRequest(new { error = "Provide a 54-character facelet string (URFDLB colors only)." });

    var psi = new ProcessStartInfo
    {
        FileName = "python3",
        ArgumentList = { "/app/python/rubik_ml_solver.py", "--json" },
        RedirectStandardInput = true,
        RedirectStandardOutput = true,
        RedirectStandardError = true
    };

    using var proc = Process.Start(psi);
    if (proc is null) return Results.Problem("Failed to start python.");

    var payload = JsonSerializer.Serialize(new PyRequest { facelets = req.Facelets, use_ml = req.UseMl ?? false });
    await proc.StandardInput.WriteAsync(payload);
    proc.StandardInput.Close();

    var stdout = await proc.StandardOutput.ReadToEndAsync();
    var stderr = await proc.StandardError.ReadToEndAsync();
    await proc.WaitForExitAsync();

    if (proc.ExitCode != 0)
        return Results.Problem($"Python error: {stderr}");

    try
    {
        using var doc = JsonDocument.Parse(stdout);
        return Results.Json(doc.RootElement.Clone());
    }
    catch
    {
        return Results.Problem("Invalid response from solver.");
    }
});

app.Run();

record SolveRequest(string Facelets, bool? UseMl);
record PyRequest { public string facelets { get; init; } = ""; public bool use_ml { get; init; } = false; }
