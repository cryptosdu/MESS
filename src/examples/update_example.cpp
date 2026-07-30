#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint32_t PROTOCOL_MAGIC = 0x51425443;
static constexpr uint16_t PROTOCOL_VERSION = 1;
static constexpr uint16_t REQUEST_FLAG_INSERT = 1 << 14;
static constexpr uint16_t REQUEST_FLAG_DELETE = 1 << 15;

struct QueryBatchRequestHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    int32_t batch_size;
    int32_t top_k;
    int32_t per_shard_pool;
    int32_t num_tables;
    int32_t sub_hash_bytes;
    int32_t body_bytes;
};

struct QueryBatchResponseHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    int32_t batch_size;
    int32_t body_bytes;
};

struct UpdateResponseHeader {
    int32_t status;
    int64_t label;
    int32_t touched_shards;
    int32_t message_bytes;
};

static bool send_all(int fd, const void* buf, size_t len) {
    const char* p = reinterpret_cast<const char*>(buf);
    size_t sent = 0;
    while (sent < len) {
        ssize_t s = ::send(fd, p + sent, len - sent, 0);
        if (s <= 0) return false;
        sent += static_cast<size_t>(s);
    }
    return true;
}

static bool recv_all(int fd, void* buf, size_t len) {
    char* p = reinterpret_cast<char*>(buf);
    size_t recvd = 0;
    while (recvd < len) {
        ssize_t r = ::recv(fd, p + recvd, len - recvd, 0);
        if (r <= 0) return false;
        recvd += static_cast<size_t>(r);
    }
    return true;
}

template <typename T>
static void append_pod(std::vector<uint8_t>& buf, const T& value) {
    const auto* p = reinterpret_cast<const uint8_t*>(&value);
    buf.insert(buf.end(), p, p + sizeof(T));
}

static uint64_t now_us() {
    using clock = std::chrono::steady_clock;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            clock::now().time_since_epoch()
        ).count()
    );
}

struct Args {
    std::string ip = "127.0.0.1";
    int port = 9090;
    std::string op = "insert";
    int dim = 768;
    int64_t label = -1;
    uint32_t seed = 1;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need_value = [&](const std::string& key) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
            return argv[++i];
        };

        if (k == "--ip") a.ip = need_value(k);
        else if (k == "--port") a.port = std::stoi(need_value(k));
        else if (k == "--op") a.op = need_value(k);
        else if (k == "--dim") a.dim = std::stoi(need_value(k));
        else if (k == "--label") a.label = std::stoll(need_value(k));
        else if (k == "--seed") a.seed = static_cast<uint32_t>(std::stoul(need_value(k)));
        else if (k == "--help" || k == "-h") {
            std::cout
                << "Usage:\n"
                << "  insert: ./example_update_client_timed --ip 10.0.0.1 --port 9090 --op insert --dim 768 --label -1 --seed 1\n"
                << "  delete: ./example_update_client_timed --ip 10.0.0.1 --port 9090 --op delete --label 123\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + k);
        }
    }
    if (a.op != "insert" && a.op != "delete") {
        throw std::runtime_error("--op must be insert or delete");
    }
    if (a.dim <= 0) {
        throw std::runtime_error("--dim must be positive");
    }
    return a;
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);

        std::vector<uint8_t> body;
        uint16_t flags = 0;

        if (args.op == "insert") {
            flags = REQUEST_FLAG_INSERT;

            std::mt19937 rng(args.seed);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            std::vector<float> vec(static_cast<size_t>(args.dim));
            for (float& x : vec) x = dist(rng);

            append_pod<int64_t>(body, args.label);
            append_pod<int32_t>(body, args.dim);
            const auto* vp = reinterpret_cast<const uint8_t*>(vec.data());
            body.insert(body.end(), vp, vp + vec.size() * sizeof(float));
        } else {
            flags = REQUEST_FLAG_DELETE;
            append_pod<int64_t>(body, args.label);
        }

        QueryBatchRequestHeader req{};
        req.magic = PROTOCOL_MAGIC;
        req.version = PROTOCOL_VERSION;
        req.flags = flags;
        req.batch_size = 1;
        req.top_k = 0;
        req.per_shard_pool = 0;
        req.num_tables = 0;
        req.sub_hash_bytes = 0;
        req.body_bytes = static_cast<int32_t>(body.size());

        const size_t req_bytes = sizeof(req) + body.size();

        uint64_t t0 = now_us();

        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) throw std::runtime_error("socket failed");

        int one = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(args.port));
        if (::inet_pton(AF_INET, args.ip.c_str(), &addr.sin_addr) != 1) {
            throw std::runtime_error("invalid ip address");
        }

        uint64_t t_connect0 = now_us();
        if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            ::close(fd);
            throw std::runtime_error("connect failed");
        }
        uint64_t t_connect1 = now_us();

        uint64_t t_send0 = now_us();
        if (!send_all(fd, &req, sizeof(req))) {
            ::close(fd);
            throw std::runtime_error("send request header failed");
        }
        if (!body.empty() && !send_all(fd, body.data(), body.size())) {
            ::close(fd);
            throw std::runtime_error("send request body failed");
        }
        uint64_t t_send1 = now_us();

        QueryBatchResponseHeader resp{};
        uint64_t t_recv0 = now_us();
        if (!recv_all(fd, &resp, sizeof(resp))) {
            ::close(fd);
            throw std::runtime_error("recv response header failed");
        }
        if (resp.magic != PROTOCOL_MAGIC || resp.version != PROTOCOL_VERSION) {
            ::close(fd);
            throw std::runtime_error("protocol mismatch in response");
        }

        std::vector<uint8_t> resp_body(static_cast<size_t>(resp.body_bytes));
        if (resp.body_bytes > 0 && !recv_all(fd, resp_body.data(), resp_body.size())) {
            ::close(fd);
            throw std::runtime_error("recv response body failed");
        }
        uint64_t t_recv1 = now_us();

        ::close(fd);
        uint64_t t1 = now_us();

        if (resp_body.size() < sizeof(UpdateResponseHeader)) {
            throw std::runtime_error("response body too small");
        }

        UpdateResponseHeader uresp{};
        std::memcpy(&uresp, resp_body.data(), sizeof(UpdateResponseHeader));

        std::string msg;
        size_t msg_off = sizeof(UpdateResponseHeader);
        if (uresp.message_bytes > 0 &&
            msg_off + static_cast<size_t>(uresp.message_bytes) <= resp_body.size()) {
            msg.assign(reinterpret_cast<const char*>(resp_body.data() + msg_off),
                       static_cast<size_t>(uresp.message_bytes));
        }

        const size_t resp_bytes = sizeof(resp) + resp_body.size();

        const double connect_ms = (t_connect1 - t_connect0) / 1000.0;
        const double send_ms = (t_send1 - t_send0) / 1000.0;
        const double recv_wait_ms = (t_recv1 - t_recv0) / 1000.0;
        const double total_ms = (t1 - t0) / 1000.0;
        const double after_connect_ms = (t1 - t_connect1) / 1000.0;

        std::cout
            << "op=" << args.op
            << " status=" << uresp.status
            << " label=" << uresp.label
            << " touched_shards=" << uresp.touched_shards
            << " message=\"" << msg << "\""
            << " connect_ms=" << connect_ms
            << " send_ms=" << send_ms
            << " recv_wait_ms=" << recv_wait_ms
            << " after_connect_ms=" << after_connect_ms
            << " total_ms=" << total_ms
            << " req_bytes=" << req_bytes
            << " resp_bytes=" << resp_bytes
            << " comm_bytes=" << (req_bytes + resp_bytes)
            << std::endl;

        return uresp.status == 0 ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
