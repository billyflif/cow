"""
Cow ID Mapping Configuration
牛ID映射配置

映射关系：真实ID（耳号）<=> 虚拟ID（耳标号）
"""

# 真实ID到虚拟ID的映射
REAL_TO_VIRTUAL_ID = {
    # "160": "000000000000000000000160",
    # "225": "000000000000000000000046",
    # "170": "000000000000000000000170",
    # "216": "000000000000000000000097",
    # "200": "000000000000000000000200",
    # "143": "000000000000000000000143",
    # "176": "000000000000000000000176",
    # "185": "000000000000000000000185",
    # "173": "000000000000000000000173",
    # "175": "000000000000000000000175",
    # "220": "000000000000000000000048",
    # "217": "000000000000000000000050",
    # "171": "000000000000000000000171",
    # "218": "000000000000000000000030",
    # "158": "000000000000000000000158",
    # "165": "000000000000000000000165",
    # "178": "000000000000000000000178",
    # "163": "000000000000000000000090",
    # "240": "000000000000000000000005",
    # "167": "000000000000000000000167",
    # "234": "000000000000000000000023",
    # "243": "000000000000000000000092",
}


def get_virtual_id_suffix(virtual_id, min_length=1):
    """
    获取虚拟ID的后缀（去除前导0）

    Args:
        virtual_id: 完整虚拟ID
        min_length: 最小保留长度（默认2位）

    Returns:
        str: 虚拟ID后缀
    """
    suffix = virtual_id.lstrip('0')
    if len(suffix) < min_length:
        suffix = virtual_id[-min_length:]
    return suffix


# 自动生成虚拟ID后缀到真实ID的映射
VIRTUAL_SUFFIX_TO_REAL_ID = {}
for real_id, virtual_id in REAL_TO_VIRTUAL_ID.items():
    # 生成多种可能的后缀格式
    suffixes = set()

    # 后2位
    suffixes.add(virtual_id[-2:])
    # 后3位
    suffixes.add(virtual_id[-3:])
    # 去除前导0的版本
    suffix = get_virtual_id_suffix(virtual_id)
    suffixes.add(suffix)

    # 将所有后缀格式都映射到真实ID
    for s in suffixes:
        if s not in VIRTUAL_SUFFIX_TO_REAL_ID:
            VIRTUAL_SUFFIX_TO_REAL_ID[s] = real_id


def virtual_to_real_id(virtual_id_suffix):
    """
    将虚拟ID后缀转换为真实ID

    Args:
        virtual_id_suffix: 虚拟ID的后缀（2-3位）

    Returns:
        str: 真实ID，如果未找到映射则返回原虚拟ID后缀
    """
    return VIRTUAL_SUFFIX_TO_REAL_ID.get(virtual_id_suffix, virtual_id_suffix)


def real_to_virtual_id(real_id):
    """
    将真实ID转换为完整虚拟ID

    Args:
        real_id: 真实ID（耳号）

    Returns:
        str: 完整虚拟ID，如果未找到映射则返回None
    """
    return REAL_TO_VIRTUAL_ID.get(real_id, None)


def add_id_mapping(real_id, virtual_id):
    """
    添加新的ID映射关系

    Args:
        real_id: 真实ID（耳号）
        virtual_id: 完整虚拟ID（耳标号）
    """
    REAL_TO_VIRTUAL_ID[real_id] = virtual_id

    # 更新反向映射
    suffixes = set()
    suffixes.add(virtual_id[-2:])
    suffixes.add(virtual_id[-3:])
    suffix = get_virtual_id_suffix(virtual_id)
    suffixes.add(suffix)

    for s in suffixes:
        VIRTUAL_SUFFIX_TO_REAL_ID[s] = real_id


if __name__ == "__main__":
    # 测试映射功能
    print("虚拟ID后缀到真实ID的映射:")
    for suffix, real_id in sorted(VIRTUAL_SUFFIX_TO_REAL_ID.items()):
        print(f"  {suffix} -> {real_id}")

    print("\n测试转换:")
    test_cases = ["23", "046", "160", "05", "5"]
    for test_id in test_cases:
        real_id = virtual_to_real_id(test_id)
        print(f"  {test_id} -> {real_id}")