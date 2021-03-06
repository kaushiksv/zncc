#ifndef SVK_CL_ERROR_MACRO_H

#define SVK_CL_ERROR_MACRO_H

#define CHECK_CL_ERROR(EMACRO)		if(enumber == EMACRO) strcat(strcat(s, ", "), #EMACRO);

#define FIND_STRINGIFY_APPEND_ALL_CL_ERRORS \
									strcat(s+2, "No matching CL_ error macro !!!");                      \
									CHECK_CL_ERROR ( CL_SUCCESS                                        ) \
									CHECK_CL_ERROR ( CL_DEVICE_NOT_FOUND                               ) \
									CHECK_CL_ERROR ( CL_DEVICE_NOT_AVAILABLE                           ) \
									CHECK_CL_ERROR ( CL_COMPILER_NOT_AVAILABLE                         ) \
									CHECK_CL_ERROR ( CL_MEM_OBJECT_ALLOCATION_FAILURE                  ) \
									CHECK_CL_ERROR ( CL_OUT_OF_RESOURCES                               ) \
									CHECK_CL_ERROR ( CL_OUT_OF_HOST_MEMORY                             ) \
									CHECK_CL_ERROR ( CL_PROFILING_INFO_NOT_AVAILABLE                   ) \
									CHECK_CL_ERROR ( CL_MEM_COPY_OVERLAP                               ) \
									CHECK_CL_ERROR ( CL_IMAGE_FORMAT_MISMATCH                          ) \
									CHECK_CL_ERROR ( CL_IMAGE_FORMAT_NOT_SUPPORTED                     ) \
									CHECK_CL_ERROR ( CL_BUILD_PROGRAM_FAILURE                          ) \
									CHECK_CL_ERROR ( CL_MAP_FAILURE                                    ) \
									CHECK_CL_ERROR ( CL_MISALIGNED_SUB_BUFFER_OFFSET                   ) \
									CHECK_CL_ERROR ( CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST      ) \
									CHECK_CL_ERROR ( CL_COMPILE_PROGRAM_FAILURE                        ) \
									CHECK_CL_ERROR ( CL_LINKER_NOT_AVAILABLE                           ) \
									CHECK_CL_ERROR ( CL_LINK_PROGRAM_FAILURE                           ) \
									CHECK_CL_ERROR ( CL_DEVICE_PARTITION_FAILED                        ) \
									CHECK_CL_ERROR ( CL_KERNEL_ARG_INFO_NOT_AVAILABLE                  ) \
									CHECK_CL_ERROR ( CL_INVALID_VALUE                                  ) \
									CHECK_CL_ERROR ( CL_INVALID_DEVICE_TYPE                            ) \
									CHECK_CL_ERROR ( CL_INVALID_PLATFORM                               ) \
									CHECK_CL_ERROR ( CL_INVALID_DEVICE                                 ) \
									CHECK_CL_ERROR ( CL_INVALID_CONTEXT                                ) \
									CHECK_CL_ERROR ( CL_INVALID_QUEUE_PROPERTIES                       ) \
									CHECK_CL_ERROR ( CL_INVALID_COMMAND_QUEUE                          ) \
									CHECK_CL_ERROR ( CL_INVALID_HOST_PTR                               ) \
									CHECK_CL_ERROR ( CL_INVALID_MEM_OBJECT                             ) \
									CHECK_CL_ERROR ( CL_INVALID_IMAGE_FORMAT_DESCRIPTOR                ) \
									CHECK_CL_ERROR ( CL_INVALID_IMAGE_SIZE                             ) \
									CHECK_CL_ERROR ( CL_INVALID_SAMPLER                                ) \
									CHECK_CL_ERROR ( CL_INVALID_BINARY                                 ) \
									CHECK_CL_ERROR ( CL_INVALID_BUILD_OPTIONS                          ) \
									CHECK_CL_ERROR ( CL_INVALID_PROGRAM                                ) \
									CHECK_CL_ERROR ( CL_INVALID_PROGRAM_EXECUTABLE                     ) \
									CHECK_CL_ERROR ( CL_INVALID_KERNEL_NAME                            ) \
									CHECK_CL_ERROR ( CL_INVALID_KERNEL_DEFINITION                      ) \
									CHECK_CL_ERROR ( CL_INVALID_KERNEL                                 ) \
									CHECK_CL_ERROR ( CL_INVALID_ARG_INDEX                              ) \
									CHECK_CL_ERROR ( CL_INVALID_ARG_VALUE                              ) \
									CHECK_CL_ERROR ( CL_INVALID_ARG_SIZE                               ) \
									CHECK_CL_ERROR ( CL_INVALID_KERNEL_ARGS                            ) \
									CHECK_CL_ERROR ( CL_INVALID_WORK_DIMENSION                         ) \
									CHECK_CL_ERROR ( CL_INVALID_WORK_GROUP_SIZE                        ) \
									CHECK_CL_ERROR ( CL_INVALID_WORK_ITEM_SIZE                         ) \
									CHECK_CL_ERROR ( CL_INVALID_GLOBAL_OFFSET                          ) \
									CHECK_CL_ERROR ( CL_INVALID_EVENT_WAIT_LIST                        ) \
									CHECK_CL_ERROR ( CL_INVALID_EVENT                                  ) \
									CHECK_CL_ERROR ( CL_INVALID_OPERATION                              ) \
									CHECK_CL_ERROR ( CL_INVALID_GL_OBJECT                              ) \
									CHECK_CL_ERROR ( CL_INVALID_BUFFER_SIZE                            ) \
									CHECK_CL_ERROR ( CL_INVALID_MIP_LEVEL                              ) \
									CHECK_CL_ERROR ( CL_INVALID_GLOBAL_WORK_SIZE                       ) \
									CHECK_CL_ERROR ( CL_INVALID_PROPERTY                               ) \
									CHECK_CL_ERROR ( CL_INVALID_IMAGE_DESCRIPTOR                       ) \
									CHECK_CL_ERROR ( CL_INVALID_COMPILER_OPTIONS                       ) \
									CHECK_CL_ERROR ( CL_INVALID_LINKER_OPTIONS                         ) \
									CHECK_CL_ERROR ( CL_INVALID_DEVICE_PARTITION_COUNT                 ) \
									CHECK_CL_ERROR ( CL_VERSION_1_0                                    ) \
									CHECK_CL_ERROR ( CL_VERSION_1_1                                    ) \
									CHECK_CL_ERROR ( CL_VERSION_1_2                                    ) \
									CHECK_CL_ERROR ( CL_FALSE                                          ) \
									CHECK_CL_ERROR ( CL_TRUE                                           ) \
									CHECK_CL_ERROR ( CL_BLOCKING                                       ) \
									CHECK_CL_ERROR ( CL_NON_BLOCKING                                   ) \
									CHECK_CL_ERROR ( CL_PLATFORM_PROFILE                               ) \
									CHECK_CL_ERROR ( CL_PLATFORM_VERSION                               ) \
									CHECK_CL_ERROR ( CL_PLATFORM_NAME                                  ) \
									CHECK_CL_ERROR ( CL_PLATFORM_VENDOR                                ) \
									CHECK_CL_ERROR ( CL_PLATFORM_EXTENSIONS                            ) \
									CHECK_CL_ERROR ( CL_DEVICE_TYPE_DEFAULT                            ) \
									CHECK_CL_ERROR ( CL_DEVICE_TYPE_CPU                                ) \
									CHECK_CL_ERROR ( CL_DEVICE_TYPE_GPU                                ) \
									CHECK_CL_ERROR ( CL_DEVICE_TYPE_ACCELERATOR                        ) \
									CHECK_CL_ERROR ( CL_DEVICE_TYPE_CUSTOM                             ) \
									CHECK_CL_ERROR ( CL_DEVICE_TYPE_ALL                                ) \
									CHECK_CL_ERROR ( CL_DEVICE_TYPE                                    ) \
									CHECK_CL_ERROR ( CL_DEVICE_VENDOR_ID                               ) \
									CHECK_CL_ERROR ( CL_DEVICE_MAX_COMPUTE_UNITS                       ) \
									CHECK_CL_ERROR ( CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS                ) \
									CHECK_CL_ERROR ( CL_DEVICE_MAX_WORK_GROUP_SIZE                     ) \
									CHECK_CL_ERROR ( CL_DEVICE_MAX_WORK_ITEM_SIZES                     ) \
									CHECK_CL_ERROR ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR             ) \
									CHECK_CL_ERROR ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT            ) \
									CHECK_CL_ERROR ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT              ) \
									CHECK_CL_ERROR ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG             ) \
									CHECK_CL_ERROR ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT            ) \
									CHECK_CL_ERROR ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE           ) \
									CHECK_CL_ERROR ( CL_DEVICE_MAX_CLOCK_FREQUENCY                     ) \
									CHECK_CL_ERROR ( CL_DEVICE_ADDRESS_BITS                            ) \
									CHECK_CL_ERROR ( CL_DEVICE_MAX_READ_IMAGE_ARGS                     ) \
									CHECK_CL_ERROR ( CL_DEVICE_MAX_WRITE_IMAGE_ARGS                    ) \
									CHECK_CL_ERROR ( CL_DEVICE_MAX_MEM_ALLOC_SIZE                      ) \
									CHECK_CL_ERROR ( CL_DEVICE_IMAGE2D_MAX_WIDTH                       ) \
									CHECK_CL_ERROR ( CL_DEVICE_IMAGE2D_MAX_HEIGHT                      ) \
									CHECK_CL_ERROR ( CL_DEVICE_IMAGE3D_MAX_WIDTH                       ) \
									CHECK_CL_ERROR ( CL_DEVICE_IMAGE3D_MAX_HEIGHT                      ) \
									CHECK_CL_ERROR ( CL_DEVICE_IMAGE3D_MAX_DEPTH                       ) \
									CHECK_CL_ERROR ( CL_DEVICE_IMAGE_SUPPORT                           ) \
									CHECK_CL_ERROR ( CL_DEVICE_MAX_PARAMETER_SIZE                      ) \
									CHECK_CL_ERROR ( CL_DEVICE_MAX_SAMPLERS                            ) \
									CHECK_CL_ERROR ( CL_DEVICE_MEM_BASE_ADDR_ALIGN                     ) \
									CHECK_CL_ERROR ( CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE                ) \
									CHECK_CL_ERROR ( CL_DEVICE_SINGLE_FP_CONFIG                        ) \
									CHECK_CL_ERROR ( CL_DEVICE_GLOBAL_MEM_CACHE_TYPE                   ) \
									CHECK_CL_ERROR ( CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE               ) \
									CHECK_CL_ERROR ( CL_DEVICE_GLOBAL_MEM_CACHE_SIZE                   ) \
									CHECK_CL_ERROR ( CL_DEVICE_GLOBAL_MEM_SIZE                         ) \
									CHECK_CL_ERROR ( CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE                ) \
									CHECK_CL_ERROR ( CL_DEVICE_MAX_CONSTANT_ARGS                       ) \
									CHECK_CL_ERROR ( CL_DEVICE_LOCAL_MEM_TYPE                          ) \
									CHECK_CL_ERROR ( CL_DEVICE_LOCAL_MEM_SIZE                          ) \
									CHECK_CL_ERROR ( CL_DEVICE_ERROR_CORRECTION_SUPPORT                ) \
									CHECK_CL_ERROR ( CL_DEVICE_PROFILING_TIMER_RESOLUTION              ) \
									CHECK_CL_ERROR ( CL_DEVICE_ENDIAN_LITTLE                           ) \
									CHECK_CL_ERROR ( CL_DEVICE_AVAILABLE                               ) \
									CHECK_CL_ERROR ( CL_DEVICE_COMPILER_AVAILABLE                      ) \
									CHECK_CL_ERROR ( CL_DEVICE_EXECUTION_CAPABILITIES                  ) \
									CHECK_CL_ERROR ( CL_DEVICE_QUEUE_PROPERTIES                        ) \
									CHECK_CL_ERROR ( CL_DEVICE_NAME                                    ) \
									CHECK_CL_ERROR ( CL_DEVICE_VENDOR                                  ) \
									CHECK_CL_ERROR ( CL_DRIVER_VERSION                                 ) \
									CHECK_CL_ERROR ( CL_DEVICE_PROFILE                                 ) \
									CHECK_CL_ERROR ( CL_DEVICE_VERSION                                 ) \
									CHECK_CL_ERROR ( CL_DEVICE_EXTENSIONS                              ) \
									CHECK_CL_ERROR ( CL_DEVICE_PLATFORM                                ) \
									CHECK_CL_ERROR ( CL_DEVICE_DOUBLE_FP_CONFIG                        ) \
									CHECK_CL_ERROR ( CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF             ) \
									CHECK_CL_ERROR ( CL_DEVICE_HOST_UNIFIED_MEMORY                     ) \
									CHECK_CL_ERROR ( CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR                ) \
									CHECK_CL_ERROR ( CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT               ) \
									CHECK_CL_ERROR ( CL_DEVICE_NATIVE_VECTOR_WIDTH_INT                 ) \
									CHECK_CL_ERROR ( CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG                ) \
									CHECK_CL_ERROR ( CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT               ) \
									CHECK_CL_ERROR ( CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE              ) \
									CHECK_CL_ERROR ( CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF                ) \
									CHECK_CL_ERROR ( CL_DEVICE_OPENCL_C_VERSION                        ) \
									CHECK_CL_ERROR ( CL_DEVICE_LINKER_AVAILABLE                        ) \
									CHECK_CL_ERROR ( CL_DEVICE_BUILT_IN_KERNELS                        ) \
									CHECK_CL_ERROR ( CL_DEVICE_IMAGE_MAX_BUFFER_SIZE                   ) \
									CHECK_CL_ERROR ( CL_DEVICE_IMAGE_MAX_ARRAY_SIZE                    ) \
									CHECK_CL_ERROR ( CL_DEVICE_PARENT_DEVICE                           ) \
									CHECK_CL_ERROR ( CL_DEVICE_PARTITION_MAX_SUB_DEVICES               ) \
									CHECK_CL_ERROR ( CL_DEVICE_PARTITION_PROPERTIES                    ) \
									CHECK_CL_ERROR ( CL_DEVICE_PARTITION_AFFINITY_DOMAIN               ) \
									CHECK_CL_ERROR ( CL_DEVICE_PARTITION_TYPE                          ) \
									CHECK_CL_ERROR ( CL_DEVICE_REFERENCE_COUNT                         ) \
									CHECK_CL_ERROR ( CL_DEVICE_PREFERRED_INTEROP_USER_SYNC             ) \
									CHECK_CL_ERROR ( CL_DEVICE_PRINTF_BUFFER_SIZE                      ) \
									CHECK_CL_ERROR ( CL_DEVICE_IMAGE_PITCH_ALIGNMENT                   ) \
									CHECK_CL_ERROR ( CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT            ) \
									CHECK_CL_ERROR ( CL_FP_DENORM                                      ) \
									CHECK_CL_ERROR ( CL_FP_INF_NAN                                     ) \
									CHECK_CL_ERROR ( CL_FP_ROUND_TO_NEAREST                            ) \
									CHECK_CL_ERROR ( CL_FP_ROUND_TO_ZERO                               ) \
									CHECK_CL_ERROR ( CL_FP_ROUND_TO_INF                                ) \
									CHECK_CL_ERROR ( CL_FP_FMA                                         ) \
									CHECK_CL_ERROR ( CL_FP_SOFT_FLOAT                                  ) \
									CHECK_CL_ERROR ( CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT               ) \
									CHECK_CL_ERROR ( CL_NONE                                           ) \
									CHECK_CL_ERROR ( CL_READ_ONLY_CACHE                                ) \
									CHECK_CL_ERROR ( CL_READ_WRITE_CACHE                               ) \
									CHECK_CL_ERROR ( CL_LOCAL                                          ) \
									CHECK_CL_ERROR ( CL_GLOBAL                                         ) \
									CHECK_CL_ERROR ( CL_EXEC_KERNEL                                    ) \
									CHECK_CL_ERROR ( CL_EXEC_NATIVE_KERNEL                             ) \
									CHECK_CL_ERROR ( CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE            ) \
									CHECK_CL_ERROR ( CL_QUEUE_PROFILING_ENABLE                         ) \
									CHECK_CL_ERROR ( CL_CONTEXT_REFERENCE_COUNT                        ) \
									CHECK_CL_ERROR ( CL_CONTEXT_DEVICES                                ) \
									CHECK_CL_ERROR ( CL_CONTEXT_PROPERTIES                             ) \
									CHECK_CL_ERROR ( CL_CONTEXT_NUM_DEVICES                            ) \
									CHECK_CL_ERROR ( CL_CONTEXT_PLATFORM                               ) \
									CHECK_CL_ERROR ( CL_CONTEXT_INTEROP_USER_SYNC                      ) \
									CHECK_CL_ERROR ( CL_DEVICE_PARTITION_EQUALLY                       ) \
									CHECK_CL_ERROR ( CL_DEVICE_PARTITION_BY_COUNTS                     ) \
									CHECK_CL_ERROR ( CL_DEVICE_PARTITION_BY_COUNTS_LIST_END            ) \
									CHECK_CL_ERROR ( CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN            ) \
									CHECK_CL_ERROR ( CL_DEVICE_AFFINITY_DOMAIN_NUMA                    ) \
									CHECK_CL_ERROR ( CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE                ) \
									CHECK_CL_ERROR ( CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE                ) \
									CHECK_CL_ERROR ( CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE                ) \
									CHECK_CL_ERROR ( CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE                ) \
									CHECK_CL_ERROR ( CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE      ) \
									CHECK_CL_ERROR ( CL_QUEUE_CONTEXT                                  ) \
									CHECK_CL_ERROR ( CL_QUEUE_DEVICE                                   ) \
									CHECK_CL_ERROR ( CL_QUEUE_REFERENCE_COUNT                          ) \
									CHECK_CL_ERROR ( CL_QUEUE_PROPERTIES                               ) \
									CHECK_CL_ERROR ( CL_MEM_READ_WRITE                                 ) \
									CHECK_CL_ERROR ( CL_MEM_WRITE_ONLY                                 ) \
									CHECK_CL_ERROR ( CL_MEM_READ_ONLY                                  ) \
									CHECK_CL_ERROR ( CL_MEM_USE_HOST_PTR                               ) \
									CHECK_CL_ERROR ( CL_MEM_ALLOC_HOST_PTR                             ) \
									CHECK_CL_ERROR ( CL_MEM_COPY_HOST_PTR                              ) \
									CHECK_CL_ERROR ( CL_MEM_HOST_WRITE_ONLY                            ) \
									CHECK_CL_ERROR ( CL_MEM_HOST_READ_ONLY                             ) \
									CHECK_CL_ERROR ( CL_MEM_HOST_NO_ACCESS                             ) \
									CHECK_CL_ERROR ( CL_MIGRATE_MEM_OBJECT_HOST                        ) \
									CHECK_CL_ERROR ( CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED           ) \
									CHECK_CL_ERROR ( CL_R                                              ) \
									CHECK_CL_ERROR ( CL_A                                              ) \
									CHECK_CL_ERROR ( CL_RG                                             ) \
									CHECK_CL_ERROR ( CL_RA                                             ) \
									CHECK_CL_ERROR ( CL_RGB                                            ) \
									CHECK_CL_ERROR ( CL_RGBA                                           ) \
									CHECK_CL_ERROR ( CL_BGRA                                           ) \
									CHECK_CL_ERROR ( CL_ARGB                                           ) \
									CHECK_CL_ERROR ( CL_INTENSITY                                      ) \
									CHECK_CL_ERROR ( CL_LUMINANCE                                      ) \
									CHECK_CL_ERROR ( CL_Rx                                             ) \
									CHECK_CL_ERROR ( CL_RGx                                            ) \
									CHECK_CL_ERROR ( CL_RGBx                                           ) \
									CHECK_CL_ERROR ( CL_DEPTH                                          ) \
									CHECK_CL_ERROR ( CL_DEPTH_STENCIL                                  ) \
									CHECK_CL_ERROR ( CL_SNORM_INT8                                     ) \
									CHECK_CL_ERROR ( CL_SNORM_INT16                                    ) \
									CHECK_CL_ERROR ( CL_UNORM_INT8                                     ) \
									CHECK_CL_ERROR ( CL_UNORM_INT16                                    ) \
									CHECK_CL_ERROR ( CL_UNORM_SHORT_565                                ) \
									CHECK_CL_ERROR ( CL_UNORM_SHORT_555                                ) \
									CHECK_CL_ERROR ( CL_UNORM_INT_101010                               ) \
									CHECK_CL_ERROR ( CL_SIGNED_INT8                                    ) \
									CHECK_CL_ERROR ( CL_SIGNED_INT16                                   ) \
									CHECK_CL_ERROR ( CL_SIGNED_INT32                                   ) \
									CHECK_CL_ERROR ( CL_UNSIGNED_INT8                                  ) \
									CHECK_CL_ERROR ( CL_UNSIGNED_INT16                                 ) \
									CHECK_CL_ERROR ( CL_UNSIGNED_INT32                                 ) \
									CHECK_CL_ERROR ( CL_HALF_FLOAT                                     ) \
									CHECK_CL_ERROR ( CL_FLOAT                                          ) \
									CHECK_CL_ERROR ( CL_UNORM_INT24                                    ) \
									CHECK_CL_ERROR ( CL_MEM_OBJECT_BUFFER                              ) \
									CHECK_CL_ERROR ( CL_MEM_OBJECT_IMAGE2D                             ) \
									CHECK_CL_ERROR ( CL_MEM_OBJECT_IMAGE3D                             ) \
									CHECK_CL_ERROR ( CL_MEM_OBJECT_IMAGE2D_ARRAY                       ) \
									CHECK_CL_ERROR ( CL_MEM_OBJECT_IMAGE1D                             ) \
									CHECK_CL_ERROR ( CL_MEM_OBJECT_IMAGE1D_ARRAY                       ) \
									CHECK_CL_ERROR ( CL_MEM_OBJECT_IMAGE1D_BUFFER                      ) \
									CHECK_CL_ERROR ( CL_MEM_TYPE                                       ) \
									CHECK_CL_ERROR ( CL_MEM_FLAGS                                      ) \
									CHECK_CL_ERROR ( CL_MEM_SIZE                                       ) \
									CHECK_CL_ERROR ( CL_MEM_HOST_PTR                                   ) \
									CHECK_CL_ERROR ( CL_MEM_MAP_COUNT                                  ) \
									CHECK_CL_ERROR ( CL_MEM_REFERENCE_COUNT                            ) \
									CHECK_CL_ERROR ( CL_MEM_CONTEXT                                    ) \
									CHECK_CL_ERROR ( CL_MEM_ASSOCIATED_MEMOBJECT                       ) \
									CHECK_CL_ERROR ( CL_MEM_OFFSET                                     ) \
									CHECK_CL_ERROR ( CL_IMAGE_FORMAT                                   ) \
									CHECK_CL_ERROR ( CL_IMAGE_ELEMENT_SIZE                             ) \
									CHECK_CL_ERROR ( CL_IMAGE_ROW_PITCH                                ) \
									CHECK_CL_ERROR ( CL_IMAGE_SLICE_PITCH                              ) \
									CHECK_CL_ERROR ( CL_IMAGE_WIDTH                                    ) \
									CHECK_CL_ERROR ( CL_IMAGE_HEIGHT                                   ) \
									CHECK_CL_ERROR ( CL_IMAGE_DEPTH                                    ) \
									CHECK_CL_ERROR ( CL_IMAGE_ARRAY_SIZE                               ) \
									CHECK_CL_ERROR ( CL_IMAGE_BUFFER                                   ) \
									CHECK_CL_ERROR ( CL_IMAGE_NUM_MIP_LEVELS                           ) \
									CHECK_CL_ERROR ( CL_IMAGE_NUM_SAMPLES                              ) \
									CHECK_CL_ERROR ( CL_ADDRESS_NONE                                   ) \
									CHECK_CL_ERROR ( CL_ADDRESS_CLAMP_TO_EDGE                          ) \
									CHECK_CL_ERROR ( CL_ADDRESS_CLAMP                                  ) \
									CHECK_CL_ERROR ( CL_ADDRESS_REPEAT                                 ) \
									CHECK_CL_ERROR ( CL_ADDRESS_MIRRORED_REPEAT                        ) \
									CHECK_CL_ERROR ( CL_FILTER_NEAREST                                 ) \
									CHECK_CL_ERROR ( CL_FILTER_LINEAR                                  ) \
									CHECK_CL_ERROR ( CL_SAMPLER_REFERENCE_COUNT                        ) \
									CHECK_CL_ERROR ( CL_SAMPLER_CONTEXT                                ) \
									CHECK_CL_ERROR ( CL_SAMPLER_NORMALIZED_COORDS                      ) \
									CHECK_CL_ERROR ( CL_SAMPLER_ADDRESSING_MODE                        ) \
									CHECK_CL_ERROR ( CL_SAMPLER_FILTER_MODE                            ) \
									CHECK_CL_ERROR ( CL_MAP_READ                                       ) \
									CHECK_CL_ERROR ( CL_MAP_WRITE                                      ) \
									CHECK_CL_ERROR ( CL_MAP_WRITE_INVALIDATE_REGION                    ) \
									CHECK_CL_ERROR ( CL_PROGRAM_REFERENCE_COUNT                        ) \
									CHECK_CL_ERROR ( CL_PROGRAM_CONTEXT                                ) \
									CHECK_CL_ERROR ( CL_PROGRAM_NUM_DEVICES                            ) \
									CHECK_CL_ERROR ( CL_PROGRAM_DEVICES                                ) \
									CHECK_CL_ERROR ( CL_PROGRAM_SOURCE                                 ) \
									CHECK_CL_ERROR ( CL_PROGRAM_BINARY_SIZES                           ) \
									CHECK_CL_ERROR ( CL_PROGRAM_BINARIES                               ) \
									CHECK_CL_ERROR ( CL_PROGRAM_NUM_KERNELS                            ) \
									CHECK_CL_ERROR ( CL_PROGRAM_KERNEL_NAMES                           ) \
									CHECK_CL_ERROR ( CL_PROGRAM_BUILD_STATUS                           ) \
									CHECK_CL_ERROR ( CL_PROGRAM_BUILD_OPTIONS                          ) \
									CHECK_CL_ERROR ( CL_PROGRAM_BUILD_LOG                              ) \
									CHECK_CL_ERROR ( CL_PROGRAM_BINARY_TYPE                            ) \
									CHECK_CL_ERROR ( CL_PROGRAM_BINARY_TYPE_NONE                       ) \
									CHECK_CL_ERROR ( CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT            ) \
									CHECK_CL_ERROR ( CL_PROGRAM_BINARY_TYPE_LIBRARY                    ) \
									CHECK_CL_ERROR ( CL_PROGRAM_BINARY_TYPE_EXECUTABLE                 ) \
									CHECK_CL_ERROR ( CL_BUILD_SUCCESS                                  ) \
									CHECK_CL_ERROR ( CL_BUILD_NONE                                     ) \
									CHECK_CL_ERROR ( CL_BUILD_ERROR                                    ) \
									CHECK_CL_ERROR ( CL_BUILD_IN_PROGRESS                              ) \
									CHECK_CL_ERROR ( CL_KERNEL_FUNCTION_NAME                           ) \
									CHECK_CL_ERROR ( CL_KERNEL_NUM_ARGS                                ) \
									CHECK_CL_ERROR ( CL_KERNEL_REFERENCE_COUNT                         ) \
									CHECK_CL_ERROR ( CL_KERNEL_CONTEXT                                 ) \
									CHECK_CL_ERROR ( CL_KERNEL_PROGRAM                                 ) \
									CHECK_CL_ERROR ( CL_KERNEL_ATTRIBUTES                              ) \
									CHECK_CL_ERROR ( CL_KERNEL_ARG_ADDRESS_QUALIFIER                   ) \
									CHECK_CL_ERROR ( CL_KERNEL_ARG_ACCESS_QUALIFIER                    ) \
									CHECK_CL_ERROR ( CL_KERNEL_ARG_TYPE_NAME                           ) \
									CHECK_CL_ERROR ( CL_KERNEL_ARG_TYPE_QUALIFIER                      ) \
									CHECK_CL_ERROR ( CL_KERNEL_ARG_NAME                                ) \
									CHECK_CL_ERROR ( CL_KERNEL_ARG_ADDRESS_GLOBAL                      ) \
									CHECK_CL_ERROR ( CL_KERNEL_ARG_ADDRESS_LOCAL                       ) \
									CHECK_CL_ERROR ( CL_KERNEL_ARG_ADDRESS_CONSTANT                    ) \
									CHECK_CL_ERROR ( CL_KERNEL_ARG_ADDRESS_PRIVATE                     ) \
									CHECK_CL_ERROR ( CL_KERNEL_ARG_ACCESS_READ_ONLY                    ) \
									CHECK_CL_ERROR ( CL_KERNEL_ARG_ACCESS_WRITE_ONLY                   ) \
									CHECK_CL_ERROR ( CL_KERNEL_ARG_ACCESS_READ_WRITE                   ) \
									CHECK_CL_ERROR ( CL_KERNEL_ARG_ACCESS_NONE                         ) \
									CHECK_CL_ERROR ( CL_KERNEL_ARG_TYPE_NONE                           ) \
									CHECK_CL_ERROR ( CL_KERNEL_ARG_TYPE_CONST                          ) \
									CHECK_CL_ERROR ( CL_KERNEL_ARG_TYPE_RESTRICT                       ) \
									CHECK_CL_ERROR ( CL_KERNEL_ARG_TYPE_VOLATILE                       ) \
									CHECK_CL_ERROR ( CL_KERNEL_WORK_GROUP_SIZE                         ) \
									CHECK_CL_ERROR ( CL_KERNEL_COMPILE_WORK_GROUP_SIZE                 ) \
									CHECK_CL_ERROR ( CL_KERNEL_LOCAL_MEM_SIZE                          ) \
									CHECK_CL_ERROR ( CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE      ) \
									CHECK_CL_ERROR ( CL_KERNEL_PRIVATE_MEM_SIZE                        ) \
									CHECK_CL_ERROR ( CL_KERNEL_GLOBAL_WORK_SIZE                        ) \
									CHECK_CL_ERROR ( CL_EVENT_COMMAND_QUEUE                            ) \
									CHECK_CL_ERROR ( CL_EVENT_COMMAND_TYPE                             ) \
									CHECK_CL_ERROR ( CL_EVENT_REFERENCE_COUNT                          ) \
									CHECK_CL_ERROR ( CL_EVENT_COMMAND_EXECUTION_STATUS                 ) \
									CHECK_CL_ERROR ( CL_EVENT_CONTEXT                                  ) \
									CHECK_CL_ERROR ( CL_COMMAND_NDRANGE_KERNEL                         ) \
									CHECK_CL_ERROR ( CL_COMMAND_TASK                                   ) \
									CHECK_CL_ERROR ( CL_COMMAND_NATIVE_KERNEL                          ) \
									CHECK_CL_ERROR ( CL_COMMAND_READ_BUFFER                            ) \
									CHECK_CL_ERROR ( CL_COMMAND_WRITE_BUFFER                           ) \
									CHECK_CL_ERROR ( CL_COMMAND_COPY_BUFFER                            ) \
									CHECK_CL_ERROR ( CL_COMMAND_READ_IMAGE                             ) \
									CHECK_CL_ERROR ( CL_COMMAND_WRITE_IMAGE                            ) \
									CHECK_CL_ERROR ( CL_COMMAND_COPY_IMAGE                             ) \
									CHECK_CL_ERROR ( CL_COMMAND_COPY_IMAGE_TO_BUFFER                   ) \
									CHECK_CL_ERROR ( CL_COMMAND_COPY_BUFFER_TO_IMAGE                   ) \
									CHECK_CL_ERROR ( CL_COMMAND_MAP_BUFFER                             ) \
									CHECK_CL_ERROR ( CL_COMMAND_MAP_IMAGE                              ) \
									CHECK_CL_ERROR ( CL_COMMAND_UNMAP_MEM_OBJECT                       ) \
									CHECK_CL_ERROR ( CL_COMMAND_MARKER                                 ) \
									CHECK_CL_ERROR ( CL_COMMAND_ACQUIRE_GL_OBJECTS                     ) \
									CHECK_CL_ERROR ( CL_COMMAND_RELEASE_GL_OBJECTS                     ) \
									CHECK_CL_ERROR ( CL_COMMAND_READ_BUFFER_RECT                       ) \
									CHECK_CL_ERROR ( CL_COMMAND_WRITE_BUFFER_RECT                      ) \
									CHECK_CL_ERROR ( CL_COMMAND_COPY_BUFFER_RECT                       ) \
									CHECK_CL_ERROR ( CL_COMMAND_USER                                   ) \
									CHECK_CL_ERROR ( CL_COMMAND_BARRIER                                ) \
									CHECK_CL_ERROR ( CL_COMMAND_MIGRATE_MEM_OBJECTS                    ) \
									CHECK_CL_ERROR ( CL_COMMAND_FILL_BUFFER                            ) \
									CHECK_CL_ERROR ( CL_COMMAND_FILL_IMAGE                             ) \
									CHECK_CL_ERROR ( CL_COMPLETE                                       ) \
									CHECK_CL_ERROR ( CL_RUNNING                                        ) \
									CHECK_CL_ERROR ( CL_SUBMITTED                                      ) \
									CHECK_CL_ERROR ( CL_QUEUED                                         ) \
									CHECK_CL_ERROR ( CL_BUFFER_CREATE_TYPE_REGION                      ) \
									CHECK_CL_ERROR ( CL_PROFILING_COMMAND_QUEUED                       ) \
									CHECK_CL_ERROR ( CL_PROFILING_COMMAND_SUBMIT                       ) \
									CHECK_CL_ERROR ( CL_PROFILING_COMMAND_START                        ) \
									CHECK_CL_ERROR ( CL_PROFILING_COMMAND_END                          ) 

#endif


									