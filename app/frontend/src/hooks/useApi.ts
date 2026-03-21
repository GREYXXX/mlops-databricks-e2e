import useSWR from "swr";

interface UseApiOptions {
  refreshInterval?: number;
}

export function useApi<T>(
  key: string | null,
  fetcher: () => Promise<T>,
  options?: UseApiOptions
) {
  const { data, error, isLoading, mutate } = useSWR<T>(key, fetcher, {
    refreshInterval: options?.refreshInterval ?? 30000,
    revalidateOnFocus: true,
    errorRetryCount: 3,
    dedupingInterval: 5000,
  });

  return { data, error, isLoading, refresh: mutate };
}
