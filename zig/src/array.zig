const std = @import("std");
const Error = error{ ZeroAllocation, OutOfMemory };

fn allocWithAlign(comptime T: type, n: usize, buffer: []u8, end_index_ptr: *usize) Error![]align(@alignOf(T)) T {
    const child_size: comptime_int = comptime @sizeOf(T);
    const align_size: comptime_int = comptime @alignOf(T);

    const byte_count: usize = blk: {
        const ov: struct { usize, u1 } = @mulWithOverflow(child_size, n);
        if (ov[1] != 0) return Error.OutOfMemory;
        if (ov[0] == 0) return Error.ZeroAllocation;
        break :blk ov[0];
    };

    const shifted_index: usize = blk: {
        const addr: usize = @intFromPtr(buffer.ptr + end_index_ptr.*);
        var ov = @addWithOverflow(addr, align_size - 1);
        if (ov[1] != 0) return Error.OutOfMemory;
        ov[0] &= ~@as(usize, align_size - 1);
        break :blk end_index_ptr.* + ov[0] - addr; // padding: usize = ov[0] - addr;
    };

    end_index_ptr.* = blk: {
        const new_end_index: usize = shifted_index + byte_count;
        if (new_end_index > buffer.len) return Error.OutOfMemory;
        break :blk new_end_index;
    };

    const ptr: [*]align(align_size) T = blk: {
        const ptr: [*]u8 = buffer.ptr + shifted_index;
        @memset(ptr[0..byte_count], undefined);
        break :blk @ptrCast(@alignCast(ptr));
    };

    return ptr[0..n];
}

test "allocWithAlign" {
    const page = std.testing.allocator;
    const buff: []u8 = try page.alloc(u8, 101);
    defer page.free(buff);

    var fba = std.heap.FixedBufferAllocator.init(buff);
    const fba_allocator = fba.allocator();

    var my_end_index: usize = 0;

    const list = [_]type{ f16, f32, f64, f80, f128 };

    inline for (list) |T| {
        const fba_ret: []T = try fba_allocator.alloc(T, 5);
        defer fba.reset();

        const my_ret: []T = try allocWithAlign(T, 5, buff, &my_end_index);
        defer my_end_index = 0;

        try std.testing.expectEqual(fba_ret.ptr, my_ret.ptr);
        try std.testing.expectEqual(fba_ret.len, my_ret.len);
        try std.testing.expectEqual(fba.end_index, my_end_index);
    }
}

pub fn Matrix(comptime T: type) type {
    if (@typeInfo(T) != .Float) @compileError("Matrix(...) only accepts float types.");
    const child_size: comptime_int = comptime @sizeOf(T);
    const slice_size: comptime_int = comptime @sizeOf(usize) * 2;

    return struct {
        allocator: std.mem.Allocator = std.heap.page_allocator,

        pub fn alloc(self: @This(), nrow: usize, ncol: usize) Error![][]T {
            const buffer: []u8 = try self.allocator.alloc(u8, nrow * ncol * child_size + nrow * slice_size);
            errdefer self.allocator.free(buffer);
            var end_index: usize = 0;
            const mat: [][]T = try allocWithAlign([]T, nrow, buffer, &end_index);
            for (mat) |*row| row.* = try allocWithAlign(T, ncol, buffer, &end_index);
            return mat;
        }

        pub fn free(self: @This(), mat: [][]T, nrow: usize, ncol: usize) void {
            const ptr: [*]u8 = @ptrCast(@alignCast(mat.ptr));
            const len: usize = nrow * ncol * child_size + nrow * slice_size;
            self.allocator.free(ptr[0..len]);
            return;
        }
    };
}

test "Matrix.allocator" {
    const MatF32 = Matrix(f32);
    {
        const Matf32_1 = MatF32{ .allocator = std.testing.allocator };
        const Matf32_2 = MatF32{ .allocator = std.testing.allocator };
        try std.testing.expect(&Matf32_1.allocator == &Matf32_2.allocator);
    }
    {
        const Matf32_1 = MatF32{ .allocator = std.heap.page_allocator };
        const Matf32_2 = MatF32{ .allocator = std.heap.page_allocator };
        try std.testing.expect(&Matf32_1.allocator == &Matf32_2.allocator);
    }
}

test "Matrix.alloc & Matrix.free" {
    const nrow = 35;
    const ncol = 17;
    const list = [_]type{ f16, f32, f64, f80, f128 };

    inline for (list) |T| {
        const MatT: Matrix(T) = .{ .allocator = std.testing.allocator };
        const mat: [][]T = try MatT.alloc(nrow, ncol);
        defer MatT.free(mat, nrow, ncol); // same as "defer allocator.free(buff);"

        var fba = blk: {
            const ptr: [*]u8 = @ptrCast(@alignCast(mat.ptr));
            const len: usize = nrow * ncol * @sizeOf(T) + nrow * @sizeOf(usize) * 2;
            break :blk std.heap.FixedBufferAllocator.init(ptr[0..len]);
        };
        const allocator = fba.allocator();

        const ref: [][]T = blk: {
            const tmp: [][]T = try allocator.alloc([]T, nrow);
            for (tmp) |*row| row.* = try allocator.alloc(T, ncol);
            break :blk tmp;
        };

        for (0..nrow) |i| {
            try std.testing.expect(&mat[i] == &ref[i]);
            for (0..ncol) |j| {
                try std.testing.expect(&mat[i][j] == &ref[i][j]);
            }
        }
    }
}
