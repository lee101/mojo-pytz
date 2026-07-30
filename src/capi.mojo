"""Bulk timezone transition lookup exposed through a small C ABI."""

from std.algorithm import parallelize
from std.sys.info import num_physical_cores


comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime I8Ptr = UnsafePointer[Int8, AnyOrigin[mut=True]]
comptime LOCAL_PARALLEL_COUNT = 100000
comptime LOCAL_MAX_WORKERS = 8


def upper_bound(values: I64Ptr, n: Int, needle: Int64) -> Int:
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if values[mid] <= needle:
            lo = mid + 1
        else:
            hi = mid
    return lo


def period_at(values: I64Ptr, n: Int, needle: Int64) -> Int:
    var pos = upper_bound(values, n, needle) - 1
    if pos < 0:
        return 0
    return pos


def resolve_utc(
    transitions: I64Ptr,
    ntransitions: Int,
    timestamps: I64Ptr,
    count: Int,
    indices: I32Ptr,
):
    if count <= 0:
        return

    var sorted = True
    for i in range(1, count):
        if timestamps[i] < timestamps[i - 1]:
            sorted = False
            break

    if sorted:
        var period = period_at(transitions, ntransitions, timestamps[0])
        indices[0] = Int32(period)
        for i in range(1, count):
            while (
                period + 1 < ntransitions
                and transitions[period + 1] <= timestamps[i]
            ):
                period += 1
            indices[i] = Int32(period)
    else:
        for i in range(count):
            indices[i] = Int32(
                period_at(transitions, ntransitions, timestamps[i])
            )


def resolve_local_one(
    transitions: I64Ptr,
    offsets: I64Ptr,
    dst_flags: I8Ptr,
    ntransitions: Int,
    wall: Int64,
    is_dst: Int8,
    max_offset: Int64,
    index_result: I32Ptr,
    status_result: I8Ptr,
    result_pos: Int,
):
    var first = -1
    var second = -1
    var start = period_at(transitions, ntransitions, wall - max_offset)
    var stop = period_at(transitions, ntransitions, wall + max_offset) + 1
    if stop > ntransitions:
        stop = ntransitions

    for period in range(start, stop):
        var utc_value = wall - offsets[period]
        if utc_value < transitions[period]:
            continue
        if (
            period + 1 < ntransitions
            and utc_value >= transitions[period + 1]
        ):
            continue
        if first < 0:
            first = period
        elif second < 0:
            second = period

    if first >= 0 and second < 0:
        index_result[result_pos] = Int32(first)
        status_result[result_pos] = 0
        return

    if first < 0:
        if is_dst < 0:
            index_result[result_pos] = -1
            status_result[result_pos] = 1
            return

        var near = period_at(transitions, ntransitions, wall)
        var gap_before = -1
        var gap_after = -1
        var gap_start = max(1, near - 2)
        var gap_stop = min(ntransitions, near + 4)
        for after in range(gap_start, gap_stop):
            var old_edge = transitions[after] + offsets[after - 1]
            var new_edge = transitions[after] + offsets[after]
            if old_edge <= wall and wall < new_edge:
                gap_before = after - 1
                gap_after = after
                break

        if gap_before < 0:
            index_result[result_pos] = Int32(near)
        elif is_dst != 0:
            index_result[result_pos] = Int32(gap_after)
        else:
            index_result[result_pos] = Int32(gap_before)
        status_result[result_pos] = 0
        return

    if is_dst < 0:
        index_result[result_pos] = -1
        status_result[result_pos] = 2
        return

    var want = Int8(1) if is_dst != 0 else Int8(0)
    var first_matches = dst_flags[first] == want
    var second_matches = dst_flags[second] == want
    if first_matches and not second_matches:
        index_result[result_pos] = Int32(first)
    elif second_matches and not first_matches:
        index_result[result_pos] = Int32(second)
    elif is_dst != 0:
        index_result[result_pos] = Int32(
            first if offsets[first] >= offsets[second] else second
        )
    else:
        index_result[result_pos] = Int32(
            first if offsets[first] <= offsets[second] else second
        )
    status_result[result_pos] = 0


def resolve_local(
    transitions: I64Ptr,
    offsets: I64Ptr,
    dst_flags: I8Ptr,
    ntransitions: Int,
    walls: I64Ptr,
    count: Int,
    is_dst: Int8,
    max_offset: Int64,
    indices: I32Ptr,
    statuses: I8Ptr,
):
    var workers = (
        min(LOCAL_MAX_WORKERS, num_physical_cores())
        if count >= LOCAL_PARALLEL_COUNT
        else 1
    )

    @parameter
    def process(worker: Int):
        var start = worker * count // workers
        var stop = (worker + 1) * count // workers
        for i in range(start, stop):
            resolve_local_one(
                transitions,
                offsets,
                dst_flags,
                ntransitions,
                walls[i],
                is_dst,
                max_offset,
                indices,
                statuses,
                i,
            )

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


@export("mptz_resolve_utc")
def mptz_resolve_utc(
    transitions_addr: Int,
    ntransitions: Int,
    timestamps_addr: Int,
    count: Int,
    indices_addr: Int,
) abi("C"):
    resolve_utc(
        I64Ptr(unsafe_from_address=transitions_addr),
        ntransitions,
        I64Ptr(unsafe_from_address=timestamps_addr),
        count,
        I32Ptr(unsafe_from_address=indices_addr),
    )


@export("mptz_resolve_local")
def mptz_resolve_local(
    transitions_addr: Int,
    offsets_addr: Int,
    dst_flags_addr: Int,
    ntransitions: Int,
    walls_addr: Int,
    count: Int,
    is_dst: Int8,
    max_offset: Int64,
    indices_addr: Int,
    statuses_addr: Int,
) abi("C"):
    resolve_local(
        I64Ptr(unsafe_from_address=transitions_addr),
        I64Ptr(unsafe_from_address=offsets_addr),
        I8Ptr(unsafe_from_address=dst_flags_addr),
        ntransitions,
        I64Ptr(unsafe_from_address=walls_addr),
        count,
        is_dst,
        max_offset,
        I32Ptr(unsafe_from_address=indices_addr),
        I8Ptr(unsafe_from_address=statuses_addr),
    )
