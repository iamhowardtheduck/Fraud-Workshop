#!/usr/bin/env python3
"""
Synthetic event generator for the `e1e-instruqt` data stream.

Produces ECS-aligned netflow / network-security records (DNS, HTTP, TLS, FTP,
SMTP/email, flow stats) matching the supplied mapping and bulk-ships them to
Elasticsearch.

Auth: basic auth (username/password) against an ephemeral VM cluster, read from
environment variables with workshop defaults:
    ELASTICSEARCH_URL       (default http://localhost:30920)
    ELASTICSEARCH_USERNAME  (default "instruqt")
    ELASTICSEARCH_PASSWORD  (default "workshops")

Usage:
    # Backfill: realistic 7-day traffic curve (diurnal + business-hours),
    # default peak ~200 EPS so it loads in ~30 min on a 2-hour VM.
    python generate_e1e_instruqt.py backfill --days 7 --peak-eps 200

    # Live: sustain ~4000 EPS continuously until Ctrl-C (matches workshop rate).
    python generate_e1e_instruqt.py live --eps 4000

    # Dry run (print sample docs, no shipping)
    python generate_e1e_instruqt.py dry-run --count 3
"""

import argparse
import math
import multiprocessing as mp
import os
import queue
import random
import signal
import sys
import threading
import time
import uuid
import warnings
from datetime import datetime, timedelta, timezone

try:
    from elasticsearch import Elasticsearch, helpers
except ImportError:
    Elasticsearch = None
    helpers = None

# Ephemeral VM may use http or a self-signed cert; quiet the TLS warning.
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

# Silence the harmless "urllib3 (2.x) doesn't match a supported version" notice
# emitted by `requests` (a transitive dep we don't actually use).
warnings.filterwarnings("ignore", message=r".*urllib3.*chardet.*")

from faker import Faker

fake = Faker()

DATA_STREAM = "e1e-instruqt"
ECS_VERSION = "8.17.0"
AGENT_VERSION = "8.12.0"

PROTOCOLS = ["dns", "http", "tls", "ftp", "smtp", "flow"]
PROTO_WEIGHTS = [0.25, 0.30, 0.20, 0.05, 0.10, 0.10]

DNS_QTYPES = ["A", "AAAA", "CNAME", "MX", "NS", "PTR", "SOA", "SRV", "TXT"]
DNS_RCODES = ["NOERROR", "NXDOMAIN", "SERVFAIL", "REFUSED"]
HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]
HTTP_STATUS = ["200", "201", "204", "301", "302", "304", "400", "401", "403", "404", "500", "502"]
MIME_TYPES = ["text/html", "application/json", "image/png", "text/css",
              "application/javascript", "application/octet-stream"]
TLS_VERSIONS = ["1.0", "1.1", "1.2", "1.3"]
TCP_FLAGS = ["S", "SA", "A", "PA", "FA", "R", "RA", "FPA"]
SENSOR_TYPES = ["yaf", "super_mediator", "pmacct", "netflow-v9"]
EXPORTERS = ["sensor-edge-01", "sensor-core-02", "sensor-dmz-03"]

# Pools to create realistic repetition / entity correlation.
# Pre-generating expensive Faker values once (companies, emails, filenames,
# certs) is both faster at runtime and more realistic — real traffic reuses
# the same senders, certs, and files rather than inventing unique ones per event.
SRC_IPS = [fake.ipv4() for _ in range(40)]
DST_IPS = [fake.ipv4() for _ in range(60)]
DOMAINS = [fake.domain_name() for _ in range(50)]
HOSTS = [f"host-{i:03d}" for i in range(1, 26)]
USER_AGENTS = [fake.user_agent() for _ in range(15)]
COMPANIES = [fake.company() for _ in range(40)]
COUNTRY_CODES = [fake.country_code() for _ in range(30)]
EMAILS = [fake.email() for _ in range(120)]
FILENAMES = [fake.file_name() for _ in range(60)]
USERNAMES = [fake.user_name() for _ in range(40)]
SUBJECTS = [fake.sentence(nb_words=6) for _ in range(80)]
CA_NAMES = ["DigiCert CA", "Let's Encrypt R3", "GlobalSign", "Sectigo RSA", "Amazon RSA 2048 M02"]
CERT_DATES = [fake.date_time_this_year().isoformat() for _ in range(50)]
ANSWER_IPS = [fake.ipv4() for _ in range(80)]
EXPORTER_IPS = [fake.ipv4() for _ in range(8)]


def now_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def epoch_millis(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def mac() -> str:
    return ":".join(f"{random.randint(0,255):02x}" for _ in range(6))


def base_doc(ts: datetime) -> dict:
    src_ip = random.choice(SRC_IPS)
    dst_ip = random.choice(DST_IPS)
    src_port = random.randint(1024, 65535)
    dst_port = random.choice([53, 80, 443, 21, 25, 8080, 22, 3389, random.randint(1, 65535)])
    duration = random.randint(1, 5_000_000)  # ns-ish
    start = ts
    end = ts + timedelta(milliseconds=random.randint(1, 30000))

    return {
        "@timestamp": now_iso(ts),
        "ecs": {"version": ECS_VERSION},
        "agent": {
            "ephemeral_id": str(uuid.uuid4()),
            "id": str(uuid.uuid4()),
            "name": random.choice(EXPORTERS),
            "type": "filebeat",
            "version": AGENT_VERSION,
        },
        "data_stream": {"dataset": DATA_STREAM, "namespace": "default"},
        "host": {"name": random.choice(HOSTS)},
        "input": {"type": "netflow"},
        "observer": {
            "name": random.choice(EXPORTERS),
            "egress": {"interface": {"id": str(random.randint(1, 8))}},
            "ingress": {"interface": {"id": str(random.randint(1, 8))}},
        },
        "labels": {
            "sensorType": random.choice(SENSOR_TYPES),
            "observation_domain_id": str(random.randint(1, 10)),
            "silkapplabel": str(dst_port),
            "ip_class_of_service": str(random.randint(0, 7)),
        },
        "source": {
            "ip": src_ip,
            "address": src_ip,
            "port": src_port,
            "mac": mac(),
            "bytes": random.randint(40, 1_500_000),
            "bytes_Reverse": random.randint(40, 1_500_000),
            "Tcp_flags": {
                "initial": random.choice(TCP_FLAGS),
                "union": random.choice(TCP_FLAGS),
            },
            "tcpSequenceNumber": str(random.randint(0, 2**32 - 1)),
        },
        "destination": {
            "ip": dst_ip,
            "address": dst_ip,
            "port": dst_port,
            "mac": mac(),
            "packets": random.randint(1, 5000),
            "Tcp_flags": {"union": random.choice(TCP_FLAGS)},
        },
        "network": {
            "iana_number": str(random.choice([6, 17, 1])),
            "packets": random.randint(1, 5000),
            "vlan": {"id": str(random.randint(1, 4094))},
        },
        "event": {
            "duration": duration,
            "start": now_iso(start),
            "end": now_iso(end),
            "Delta_ms": random.randint(0, 30000),
        },
        "netflowId": str(uuid.uuid4()),
        "flowStartDate": ts.strftime("%Y-%m-%d"),
        "retry": False,
    }


def add_dns(doc: dict):
    qname = random.choice(DOMAINS)
    doc["event"]["category"] = "network"
    doc["event"]["type"] = "protocol"
    doc["dns"] = {
        "ID": str(random.randint(1, 65535)),
        "id": str(random.randint(1, 65535)),
        "QName": qname,
        "QRType": random.choice(DNS_QTYPES),
        "response_code": random.choice(DNS_RCODES),
        "question": {"name": qname},
        "A": random.choice(ANSWER_IPS) if random.random() > 0.3 else None,
        "TTL": str(random.choice([60, 300, 3600, 86400])),
        "answers": {"ttl": str(random.choice([60, 300, 3600]))},
    }
    doc["dns"] = {k: v for k, v in doc["dns"].items() if v is not None}


def add_http(doc: dict):
    doc["event"]["category"] = "web"
    doc["event"]["type"] = "access"
    ua = random.choice(USER_AGENTS)
    doc["http"] = {
        "Host": random.choice(DOMAINS),
        "version": random.choice(["1.0", "1.1", "2"]),
        "Connection": random.choice(["keep-alive", "close"]),
        "request": {
            "method": random.choice(HTTP_METHODS),
            "referer": f"https://{random.choice(DOMAINS)}/",
            "Header": {
                "accept": "text/html,application/xhtml+xml",
                "host": random.choice(DOMAINS),
                "x-Forwarded-For": random.choice(SRC_IPS),
            },
        },
        "response": {
            "status_code": random.choice(HTTP_STATUS),
            "mime_type": random.choice(MIME_TYPES),
            "Header": {
                "server": random.choice(["nginx", "apache", "cloudflare", "envoy"]),
                "bytes": str(random.randint(100, 500000)),
            },
        },
        "user_agent": {"original": ua},
    }
    doc["user_agent"] = {"original": ua}


def add_tls(doc: dict):
    doc["event"]["category"] = "network"
    doc["event"]["type"] = "connection"
    sni = random.choice(DOMAINS)
    doc["tls"] = {
        "version": random.choice(TLS_VERSIONS),
        "Record_version": random.choice([769, 770, 771, 772]),
        "ServerCipher": random.randint(1, 60000),
        "Compression_method": random.choice([0, 1]),
        "client": {
            "server_name": sni,
            "x509": {
                "subject": {"common_name": sni, "organization": random.choice(COMPANIES),
                            "country": random.choice(COUNTRY_CODES)},
                "issuer": {"common_name": random.choice(CA_NAMES),
                           "organization": random.choice(COMPANIES)},
                "not_before": random.choice(CERT_DATES),
                "not_after": random.choice(CERT_DATES),
                "serial_number": uuid.uuid4().hex,
                "public_key_algorithm": random.choice(["RSA", "EC"]),
                "public_key_size": str(random.choice([2048, 256, 384, 4096])),
            },
        },
        "sslClientJA3": uuid.uuid4().hex,
        "sslServerJA3S": uuid.uuid4().hex,
    }


def add_ftp(doc: dict):
    doc["event"]["category"] = "file"
    doc["event"]["type"] = "access"
    doc["ftp"] = {
        "User_name": random.choice(USERNAMES),
        "User_password": "********",
        "Data_type": random.choice(["ASCII", "Binary", "Image"]),
        "Response_code": random.choice(["200", "230", "331", "425", "530"]),
        "Return": random.choice(["RETR", "STOR", "LIST", "PASV", "USER"]),
    }


def add_email(doc: dict):
    doc["event"]["category"] = "email"
    doc["event"]["type"] = "info"
    frm = random.choice(EMAILS)
    to = random.choice(EMAILS)
    doc["email"] = {
        "from": {"address": frm},
        "to": {"address": to},
        "subject": random.choice(SUBJECTS),
        "content_type": random.choice(["text/plain", "text/html", "multipart/mixed"]),
        "Size": str(random.randint(500, 5_000_000)),
        "Hello": random.choice(DOMAINS),
        "Response": random.choice(["250 OK", "550 No such user", "421 Service unavailable"]),
        "attachments": {"file": {"name": random.choice(FILENAMES)}} if random.random() > 0.6 else {},
    }


def add_flow_stats(doc: dict):
    doc["event"]["category"] = "network"
    doc["event"]["type"] = "connection"
    doc["stats"] = {
        "packetTotalCount": random.randint(1, 1_000_000),
        "droppedPacketTotalCount": random.randint(0, 500),
        "ignoredPacketTotalCount": random.randint(0, 200),
        "exportedFlowTotalCount": random.randint(1, 100000),
        "flowTablePeakCount": random.randint(1, 50000),
        "meanFlowRate": random.randint(0, 100000),
        "meanPacketRate": random.randint(0, 1_000_000),
        "exporterName": random.choice(EXPORTERS),
        "exporterIPv4Address": random.choice(EXPORTER_IPS),
        "observationDomainId": random.randint(1, 10),
        "exportingProcessId": random.randint(1000, 9999),
    }


PROTO_FUNCS = {
    "dns": add_dns,
    "http": add_http,
    "tls": add_tls,
    "ftp": add_ftp,
    "smtp": add_email,
    "flow": add_flow_stats,
}


def make_doc(ts: datetime) -> dict:
    doc = base_doc(ts)
    proto = random.choices(PROTOCOLS, weights=PROTO_WEIGHTS, k=1)[0]
    PROTO_FUNCS[proto](doc)
    doc["tags"] = [proto, "synthetic", "instruqt"]
    return doc


def gen_docs(count: int, span_minutes: int):
    """Legacy helper: `count` docs spread across the last `span_minutes`."""
    start = datetime.now(timezone.utc) - timedelta(minutes=span_minutes)
    for i in range(count):
        offset = (span_minutes * 60) * (i / max(count, 1))
        ts = start + timedelta(seconds=offset + random.uniform(0, 1))
        yield make_doc(ts)


# --------------------------------------------------------------------------- #
#  Traffic-shape model (shared by backfill)                                    #
# --------------------------------------------------------------------------- #

def eps_at(ts: datetime, peak_eps: float, trough_eps: float) -> float:
    """Events-per-second at a given timestamp following a realistic curve:
    diurnal sine peaking ~13:00, a business-hours boost (09-17 weekdays),
    and weekend dampening."""
    hour = ts.hour + ts.minute / 60.0
    weekend = ts.weekday() >= 5
    day_scale = 0.4 if weekend else 1.0
    diurnal = (math.sin((hour - 7) / 24 * 2 * math.pi) + 1) / 2     # 0..1
    biz = 1.0 + (0.6 if (9 <= hour <= 17 and not weekend) else 0.0)
    eps = trough_eps + (peak_eps - trough_eps) * diurnal * biz * day_scale
    return max(trough_eps, min(eps, peak_eps))


def backfill_docs(days: int, peak_eps: float, trough_eps: float, bucket_s: int = 60):
    """Yield docs for the past `days`, generating each `bucket_s`-second bucket
    with a count derived from the traffic curve at that bucket's time."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    t = start
    while t < now:
        rate = eps_at(t, peak_eps, trough_eps)
        n = max(0, int(random.gauss(rate * bucket_s, rate * bucket_s * 0.1)))
        for _ in range(n):
            ts = t + timedelta(seconds=random.uniform(0, bucket_s))
            yield make_doc(ts)
        t += timedelta(seconds=bucket_s)


def estimate_backfill(days: int, peak_eps: float, trough_eps: float) -> int:
    total = 0
    now = datetime.now(timezone.utc)
    t = now - timedelta(days=days)
    while t < now:
        total += eps_at(t, peak_eps, trough_eps) * 3600
        t += timedelta(hours=1)
    return int(total)


# --------------------------------------------------------------------------- #
#  Elasticsearch connection                                                    #
# --------------------------------------------------------------------------- #

def connect(announce=True):
    if Elasticsearch is None:
        sys.exit("elasticsearch package not installed. Run: pip install elasticsearch")
    es_url = os.environ.get("ELASTICSEARCH_URL", "http://localhost:30920")
    username = os.environ.get("ELASTICSEARCH_USERNAME", "instruqt")
    password = os.environ.get("ELASTICSEARCH_PASSWORD", "workshops")
    es = Elasticsearch(
        es_url,
        basic_auth=(username, password),
        verify_certs=False,
    ).options(request_timeout=60, retry_on_timeout=True, max_retries=3)
    if announce:
        try:
            info = es.info()
        except Exception as e:
            msg = str(e)
            if "401" in msg or "authenticate" in msg:
                sys.exit(
                    f"Authentication failed for user '{username}' at {es_url}.\n"
                    f"  Verify credentials, e.g.:\n"
                    f"    curl -u {username}:<password> {es_url}\n"
                    f"  then set ELASTICSEARCH_USERNAME / ELASTICSEARCH_PASSWORD.")
            sys.exit(f"Could not reach Elasticsearch at {es_url}: {e}")
        print(f"Connected to {es_url} | cluster '{info['cluster_name']}' "
              f"| version {info['version']['number']} | user '{username}'", flush=True)
    return es


def to_action(index, doc):
    # `create` op -> required for append-only data streams
    return {"_op_type": "create", "_index": index, "_source": doc}


# --------------------------------------------------------------------------- #
#  Parallel bulk shipper (used by both backfill and live)                      #
# --------------------------------------------------------------------------- #

class Shipper:
    """Fan a doc stream out to N worker threads doing parallel bulk indexing.
    Generation runs on the main thread; workers pull from a bounded queue."""

    def __init__(self, es, index, workers=4, batch=2000, queue_batches=8):
        self.es = es
        self.index = index
        self.batch = batch
        self.q = queue.Queue(maxsize=queue_batches)
        self.workers = [threading.Thread(target=self._worker, daemon=True)
                        for _ in range(workers)]
        self.lock = threading.Lock()
        self.ok = 0
        self.failed = 0
        self.first_error = None
        self.stop = threading.Event()

    def start(self):
        for w in self.workers:
            w.start()

    def _worker(self):
        while not self.stop.is_set():
            try:
                chunk = self.q.get(timeout=1)
            except queue.Empty:
                continue
            if chunk is None:
                self.q.task_done()
                break
            actions = [to_action(self.index, d) for d in chunk]
            try:
                ok, errs = helpers.bulk(self.es, actions, raise_on_error=False)
                with self.lock:
                    self.ok += ok
                    if errs:
                        self.failed += len(errs)
                        if self.first_error is None:
                            self.first_error = errs[0]
            except Exception as e:
                with self.lock:
                    self.failed += len(actions)
                    if self.first_error is None:
                        self.first_error = repr(e)
            finally:
                self.q.task_done()

    def submit(self, chunk):
        self.q.put(chunk)

    def close(self):
        for _ in self.workers:
            self.q.put(None)
        for w in self.workers:
            w.join()


def chunked(it, size):
    chunk = []
    for x in it:
        chunk.append(x)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# --------------------------------------------------------------------------- #
#  Modes                                                                        #
# --------------------------------------------------------------------------- #

def run_backfill(args):
    est = estimate_backfill(args.days, args.peak_eps, args.trough_eps)
    procs = args.procs
    print(f"Backfill plan: {args.days} days, peak {args.peak_eps} EPS, "
          f"trough {args.trough_eps} EPS  ->  ~{est:,} events", flush=True)
    print(f"Spawning {procs} generator processes x {args.workers} ship threads each",
          flush=True)
    connect()  # validate connectivity / print banner once from the parent

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.days)
    total_s = (now - start).total_seconds()
    slice_s = total_s / procs

    result_q = mp.Queue()
    jobs = []
    for i in range(procs):
        w_start = start + timedelta(seconds=slice_s * i)
        w_end = start + timedelta(seconds=slice_s * (i + 1))
        p = mp.Process(target=_backfill_worker,
                       args=(i, w_start, w_end, args, result_q))
        p.start()
        jobs.append(p)

    t0 = time.time()
    ok = failed = 0
    first_error = None
    for _ in jobs:
        r = result_q.get()
        ok += r["ok"]
        failed += r["failed"]
        first_error = first_error or r["first_error"]
    for p in jobs:
        p.join()

    elapsed = time.time() - t0
    print(f"Backfill done: indexed {ok:,}, failed {failed:,} in {elapsed:.0f}s "
          f"({ok/elapsed:,.0f} docs/s) into '{args.index}'", flush=True)
    if first_error:
        print(f"  first error: {first_error}", flush=True)


def _backfill_worker(wid, w_start, w_end, args, result_q):
    # Each process re-seeds RNG so slices aren't identical.
    random.seed(os.getpid() ^ int(time.time() * 1000) ^ wid)
    es = connect(announce=False)
    shipper = Shipper(es, args.index, workers=args.workers, batch=args.batch)
    shipper.start()
    sent = 0
    t0 = time.time()
    for chunk in chunked(_backfill_slice(w_start, w_end, args.peak_eps,
                                         args.trough_eps), args.batch):
        shipper.submit(chunk)
        sent += len(chunk)
        if wid == 0 and sent % (args.batch * 50) == 0:
            el = time.time() - t0
            print(f"  [p0] {sent:,} generated ({sent/el:,.0f} docs/s/proc)", flush=True)
    shipper.close()
    result_q.put({"ok": shipper.ok, "failed": shipper.failed,
                  "first_error": shipper.first_error})


def _backfill_slice(w_start, w_end, peak_eps, trough_eps, bucket_s=60):
    """Like backfill_docs but bounded to [w_start, w_end) for one process."""
    t = w_start
    while t < w_end:
        rate = eps_at(t, peak_eps, trough_eps)
        n = max(0, int(random.gauss(rate * bucket_s, rate * bucket_s * 0.1)))
        for _ in range(n):
            ts = t + timedelta(seconds=random.uniform(0, bucket_s))
            yield make_doc(ts)
        t += timedelta(seconds=bucket_s)


def run_live(args):
    procs = args.procs
    per_proc_eps = args.eps / procs
    print(f"Live mode: targeting ~{args.eps:,.0f} EPS into '{args.index}' "
          f"across {procs} processes (~{per_proc_eps:,.0f} EPS each). "
          f"Press Ctrl-C to stop.", flush=True)
    connect()  # banner + connectivity check from parent

    stop = mp.Event()
    result_q = mp.Queue()
    jobs = []
    for i in range(procs):
        p = mp.Process(target=_live_worker, args=(i, per_proc_eps, args, stop, result_q))
        p.start()
        jobs.append(p)

    # Parent handles Ctrl-C and signals all children to drain.
    def handle_sigint(signum, frame):
        print("\nStopping (draining)...", flush=True)
        stop.set()
    signal.signal(signal.SIGINT, handle_sigint)

    for p in jobs:
        p.join()

    total = ok = failed = 0
    first_error = None
    while not result_q.empty():
        r = result_q.get()
        total += r["total"]; ok += r["ok"]; failed += r["failed"]
        first_error = first_error or r["first_error"]
    print(f"Live stopped: {total:,} generated, indexed {ok:,}, failed {failed:,}",
          flush=True)
    if first_error:
        print(f"  first error: {first_error}", flush=True)


def _live_worker(wid, eps, args, stop, result_q):
    random.seed(os.getpid() ^ int(time.time() * 1000) ^ wid)
    # Children ignore SIGINT directly; parent sets the shared stop event.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    es = connect(announce=False)
    shipper = Shipper(es, args.index, workers=args.workers, batch=args.batch)
    shipper.start()

    tick = 0.25
    per_tick = max(1, int(eps * tick))
    t0 = time.time()
    next_t = t0
    total = 0
    last_report = t0
    while not stop.is_set():
        now = time.time()
        batch = [make_doc(datetime.now(timezone.utc)) for _ in range(per_tick)]
        for c in chunked(batch, args.batch):
            shipper.submit(c)
        total += len(batch)
        next_t += tick
        sleep = next_t - time.time()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_t = time.time()
        if wid == 0 and now - last_report >= 5:
            el = now - t0
            # report aggregate estimate (this proc x procs)
            print(f"  live[p0]: {total:,} sent this proc, "
                  f"~{total/el:,.0f} EPS/proc, indexed {shipper.ok:,}, "
                  f"failed {shipper.failed:,}", flush=True)
            last_report = now

    shipper.close()
    result_q.put({"total": total, "ok": shipper.ok, "failed": shipper.failed,
                  "first_error": shipper.first_error})


def run_dry(args):
    import json
    for d in gen_docs(min(args.count, 5), 60):
        print(json.dumps(d, indent=2))


# --------------------------------------------------------------------------- #
#  CLI                                                                          #
# --------------------------------------------------------------------------- #

def main():
    default_procs = max(1, (os.cpu_count() or 4) - 2)
    p = argparse.ArgumentParser(description="Event generator for the e1e-instruqt data stream")
    p.add_argument("--index", default=DATA_STREAM, help="target data stream / index")
    p.add_argument("--batch", type=int, default=2000, help="bulk batch size")
    p.add_argument("--procs", type=int, default=default_procs,
                   help=f"generator processes (default {default_procs}: cores-2)")
    p.add_argument("--workers", type=int, default=3,
                   help="bulk ship threads PER process (default 3)")
    sub = p.add_subparsers(dest="mode", required=True)

    b = sub.add_parser("backfill", help="generate a realistic N-day history")
    b.add_argument("--days", type=int, default=7)
    b.add_argument("--peak-eps", type=float, default=200,
                   help="peak events/sec at midday weekdays (default 200, ~56M/7d)")
    b.add_argument("--trough-eps", type=float, default=8,
                   help="overnight floor events/sec (default 8)")
    b.set_defaults(func=run_backfill)

    l = sub.add_parser("live", help="sustain a target EPS until Ctrl-C")
    l.add_argument("--eps", type=float, default=4000, help="target events/sec (default 4000)")
    l.set_defaults(func=run_live)

    d = sub.add_parser("dry-run", help="print sample docs, do not ship")
    d.add_argument("--count", type=int, default=3)
    d.set_defaults(func=run_dry)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
