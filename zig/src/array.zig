const std = @import("std");
const Error = std.mem.Allocator.Error;

// already comptime scope
const slice_al: comptime_int = @alignOf([]f64);
const child_al: comptime_int = @alignOf(f64);
const slice_sz: comptime_int = @sizeOf(usize) * 2;
const child_sz: comptime_int = @sizeOf(f64);

pub const Matrix = struct {
    allocator: std.mem.Allocator = std.heap.page_allocator,

    pub fn alloc(self: Matrix, nrow: usize, ncol: usize) ![][]f64 {
        const buff: []u8 = try self.allocator.alloc(u8, nrow * ncol * child_sz + nrow * slice_sz);
        errdefer self.allocator.free(buff);

        const mat: [][]f64 = blk: {
            const ptr: [*]align(slice_al) []f64 = @ptrCast(@alignCast(buff.ptr));
            break :blk ptr[0..nrow];
        };

        const chunk_sz: usize = ncol * child_sz;
        var padding: usize = nrow * slice_sz;

        for (mat) |*row| {
            row.* = blk: {
                const ptr: [*]align(child_al) f64 = @ptrCast(@alignCast(buff.ptr + padding));
                break :blk ptr[0..ncol];
            };
            padding += chunk_sz;
        }
        return mat;
    }

    pub fn free(self: Matrix, mat: [][]f64, nrow: usize, ncol: usize) void {
        const ptr: [*]u8 = @ptrCast(@alignCast(mat.ptr));
        const len: usize = nrow * ncol * child_sz + nrow * slice_sz;
        self.allocator.free(ptr[0..len]);
        return;
    }
};

pub const Vector = struct {
    allocator: std.mem.Allocator = std.heap.page_allocator,

    pub fn alloc(self: Vector, n: usize) Error![]f64 {
        return try self.allocator.alloc(f64, n);
    }

    pub fn free(self: Vector, vec: []f64) void {
        self.allocator.free(vec);
        return;
    }
};

test "Matrix.alloc & Matrix.free" {
    const nrow = 35;
    const ncol = 17;

    const MatT: Matrix = .{ .allocator = std.testing.allocator };
    const mat: [][]f64 = try MatT.alloc(nrow, ncol);
    defer MatT.free(mat, nrow, ncol);

    var fba = blk: {
        const ptr: [*]u8 = @ptrCast(@alignCast(mat.ptr));
        const len: usize = nrow * ncol * child_sz + nrow * slice_sz;
        break :blk std.heap.FixedBufferAllocator.init(ptr[0..len]);
    };
    const allocator = fba.allocator();

    const ref: [][]f64 = blk: {
        const tmp: [][]f64 = try allocator.alloc([]f64, nrow);
        for (tmp) |*row| row.* = try allocator.alloc(f64, ncol);
        break :blk tmp;
    };

    for (0..nrow) |i| {
        try std.testing.expect(&mat[i] == &ref[i]);
        for (0..ncol) |j| {
            try std.testing.expect(&mat[i][j] == &ref[i][j]);
        }
    }
}
