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
        // -u = unbuffered stdout/stderr, so we don't lose the traceback
        ArgumentList = { "-u", "/app/python/rubik_ml_solver.py", "--json" },
        RedirectStandardInput = true,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false
    };

    // helpful env for safety
    psi.Environment["PYTHONUNBUFFERED"] = "1";

    using var proc = Process.Start(psi);
    if (proc is null) return Results.Problem("Failed to start python.");

    var payload = JsonSerializer.Serialize(new PyRequest { facelets = req.Facelets, use_ml = req.UseMl ?? false });
    await proc.StandardInput.WriteAsync(payload);
    proc.StandardInput.Close();

    var stdoutTask = proc.StandardOutput.ReadToEndAsync();
    var stderrTask = proc.StandardError.ReadToEndAsync();

    await Task.WhenAll(stdoutTask, stderrTask);
    var stdout = stdoutTask.Result;
    var stderr = stderrTask.Result;

    await proc.WaitForExitAsync();

    // If Python returned JSON on stdout, try to surface that, even on non-zero exit.
    if (!string.IsNullOrWhiteSpace(stdout))
    {
        try
        {
            using var doc = JsonDocument.Parse(stdout);
            // If it contains an "error" key, map to 400; else forward as normal
            if (doc.RootElement.TryGetProperty("error", out var errEl))
            {
                return Results.BadRequest(new { error = errEl.GetString() });
            }
            return Results.Json(doc.RootElement.Clone());
        }
        catch
        {
            // fall through to stderr handling
        }
    }

    // No usable stdout -> surface stderr (or a generic message)
    var msg = string.IsNullOrWhiteSpace(stderr) ? "Unknown Python error (stderr was empty)" : stderr;
    return Results.Problem($"Python error: {msg}");
});

app.Run();

record SolveRequest(string Facelets, bool? UseMl);
record PyRequest { public string facelets { get; init; } = ""; public bool use_ml { get; init; } = false; }
