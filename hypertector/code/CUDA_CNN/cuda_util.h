#include <cuda_runtime.h>

inline void check_call ( cudaError_t error, char const * file, size_t line, char const * call ) {
		if ( cudaSuccess == error ) return;
			fprintf ( stderr, "%s failed in \"%s:%lu\": %s\n", call, file, line, cudaGetErrorString ( error ) );
				exit ( error );
}

#define SAFE_CALL(call) check_call ( ( call ), __FILE__, __LINE__, #call )

