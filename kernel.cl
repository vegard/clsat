#define LOCAL_SIZE_0 256

typedef uchar u8;
typedef ushort u16;
typedef uint u32;
typedef ulong u64;

struct thread_state {
	u32 rnd;
	u32 nr_sat_clauses;
};

struct clause {
	u16 literals[4];
};

inline u32 bits(u32 rnd, const unsigned int i, const unsigned int n)
{
	return (rnd >> i) & ((1U << n) - 1);
}

u8 get_value(__global u8 *values, u16 lit)
{
	bool sign = lit & 1;
	u16 var = lit >> 1;

	return values[var * LOCAL_SIZE_0 + get_local_id(0)] ^ sign;
}

void set_value(__global u8 *values, u16 lit, u8 value)
{
	bool sign = lit & 1;
	u16 var = lit >> 1;

	values[var * LOCAL_SIZE_0 + get_local_id(0)] = value ^ sign;
}

__kernel void search(__global struct thread_state *states,
	__global u8 *values,
	unsigned int nr_clauses, __global const struct clause *clauses,
	unsigned int clause_i, unsigned int n)
{
	/* Load per-thread state */
	struct thread_state state = states[get_global_id(0)];

	__local struct clause clause_cache[LOCAL_SIZE_0];
	unsigned int clause_cache_i = 0;

	/* Initial cache fill; each thread loads a part of the cache from
	 * global memory to local memory using a single coalesced transfer
	 * as long as each clause is smaller than or equal to 128 bits. */
	clause_cache[get_local_id(0)] = clauses[get_local_id(0)];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (unsigned int i = 0; i < n; ++i) {
		/* We use a per-thread pseudo-random number generator. This particular
		 * algorithm is REALLY POOR! It doesn't matter so much, however. It is
		 * used for tie-breaking in order to prevent us from getting stuck in
		 * an infinite loop. It should be really fast, however. */
		state.rnd ^= ((state.rnd << 4) | (state.rnd >> 17)) + 0xcafebabe;

		struct clause c = clause_cache[clause_cache_i];
		u8 v0 = get_value(values, c.literals[0]);
		u8 v1 = get_value(values, c.literals[1]);
		u8 v2 = get_value(values, c.literals[2]);
		u8 v3 = get_value(values, c.literals[3]);
		u8 satisfied = v0 | v1 | v2 | v3;

		/* Pick an unsatisfied literal to flip (but only if it is not
		 * variable 0, which is always false. Each literal is weighted
		 * by a random number for tie-breaking between false literals,
		 * and the literal with the greatest weight is flipped. */
		u8 w0 = (!v0 && c.literals[0]) * (1 + bits(state.rnd,  0, 4));
		u8 w1 = (!v1 && c.literals[1]) * (1 + bits(state.rnd,  4, 4));
		u8 w2 = (!v2 && c.literals[2]) * (1 + bits(state.rnd,  8, 4));
		u8 w3 = (!v3 && c.literals[3]) * (1 + bits(state.rnd, 12, 4));

		u8 j = select(0, 1, w0 < w1);
		u8 w = max(w0, w1);

		j = select(j, 2, w < w2);
		w = max(w, w2);

		j = select(j, 3, w < w3);
		w = max(w, w3);

		/* Flip the right value. We flip if and only if:
		 *  1. the clause was not satisfied, AND
		 *  2. the literal is the chosen unsatisfied literal */
		/* XXX: We could perhaps be a bit more efficient by retrieving
		 * the literal's value and the variable's value separately. */
		set_value(values, c.literals[0], v0 ^ (!satisfied && j == 0));
		set_value(values, c.literals[1], v1 ^ (!satisfied && j == 1));
		set_value(values, c.literals[2], v2 ^ (!satisfied && j == 2));
		set_value(values, c.literals[3], v3 ^ (!satisfied && j == 3));

		/* If the clause was already satisfied, increment nr_sat_clauses;
		 * otherwise, reset it to 0. */
		state.nr_sat_clauses = select(0, state.nr_sat_clauses + 1, satisfied);

		/* Refill cache if needed */
		if (++clause_cache_i == LOCAL_SIZE_0) {
			clause_cache_i = 0;

			clause_i += get_local_size(0);
			if (clause_i == nr_clauses)
				clause_i = 0;

			clause_cache[get_local_id(0)] = clauses[clause_i + get_local_id(0)];
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}

	/* Write back per-thread state */
	states[get_global_id(0)] = state;
}
