import os
import yaml
import re
from typing import Dict, List, Any, Optional
from pathlib import Path


class PolicyEvaluator:
    """
    PolicyファイルのIF条件を評価し、条件に一致しないルールからbest_policyを生成するクラス
    """
    
    def __init__(self, base_path: str = "info"):
        """
        初期化
        
        Args:
            base_path: infoファイルのベースパス
        """
        self.base_path = Path(base_path)
        self.policy_dir = Path("policy")
        
    def load_policy_file(self, role: str) -> Dict[str, Any]:
        """
        役職に応じたpolicyファイルを読み込む
        
        Args:
            role: エージェントの役職
            
        Returns:
            policyファイルの内容
        """
        policy_file = self.policy_dir / f"{role.lower()}.yml"
        
        if not policy_file.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_file}")
            
        with open(policy_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_info_file(self, file_path: str) -> Dict[str, Any]:
        """
        infoファイル（status.yml, analysis.yml, my_log.yml）を読み込む
        
        Args:
            file_path: ファイルパス
            
        Returns:
            ファイルの内容（存在しない場合は空辞書）
        """
        if not os.path.exists(file_path):
            return {}
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    
    def replace_placeholders(self, path: str, game_name: str, agent_name: str) -> str:
        """
        パス内のプレースホルダーを実際の値に置換
        
        Args:
            path: プレースホルダーを含むパス
            game_name: ゲーム名
            agent_name: エージェント名
            
        Returns:
            置換後のパス
        """
        return path.replace("{game_name}", game_name).replace("{agent_name}", agent_name)
    
    def evaluate_file_check(self, check_info: Dict[str, str], game_name: str, agent_name: str) -> bool:
        """
        個別のfile_checkを評価
        
        Args:
            check_info: file_check情報（fileとcheckキーを含む）
            game_name: ゲーム名
            agent_name: エージェント名
            
        Returns:
            条件が満たされているかどうか
        """
        # パスの最初の/を削除してから置換
        file_path_template = check_info["file"]
        if file_path_template.startswith("/info/"):
            file_path_template = file_path_template[6:]  # "/info/"を削除
        
        file_path = self.replace_placeholders(file_path_template, game_name, agent_name)
        full_path = self.base_path / file_path
        check_condition = check_info["check"]
        
        # ファイルを読み込み
        data = self.load_info_file(str(full_path))
        
        if not data:
            # ファイルが存在しないか空の場合の処理
            return "存在しない" in check_condition or "ない" in check_condition
        
        # チェック条件の評価ロジック
        return self._evaluate_check_condition(data, check_condition, agent_name)
    
    def _evaluate_check_condition(self, data: Dict[str, Any], condition: str, agent_name: str) -> bool:
        """
        チェック条件を評価する内部メソッド
        
        Args:
            data: ファイルの内容
            condition: チェック条件
            agent_name: エージェント名
            
        Returns:
            条件が満たされているかどうか
        """
        # 条件の置換（{agent_name}を実際の名前に）
        condition = condition.replace("{agent_name}", agent_name)
        
        # 各種条件パターンの評価
        if "存在しない" in condition or "ない" in condition:
            return self._check_not_exists(data, condition)
        elif "存在" in condition:
            return self._check_exists(data, condition)
        elif "AND" in condition:
            return self._check_and_condition(data, condition)
        elif "OR" in condition:
            return self._check_or_condition(data, condition)
        elif "=" in condition:
            return self._check_equality(data, condition)
        elif ">" in condition or "<" in condition:
            return self._check_comparison(data, condition)
        else:
            # 基本的なキーワード検索
            return self._check_keyword(data, condition)
    
    def _check_not_exists(self, data: Dict[str, Any], condition: str) -> bool:
        """存在しない条件をチェック"""
        keywords = self._extract_keywords(condition)
        return not any(self._search_in_data(data, keyword) for keyword in keywords)
    
    def _check_exists(self, data: Dict[str, Any], condition: str) -> bool:
        """存在する条件をチェック"""
        keywords = self._extract_keywords(condition)
        return any(self._search_in_data(data, keyword) for keyword in keywords)
    
    def _check_and_condition(self, data: Dict[str, Any], condition: str) -> bool:
        """AND条件をチェック"""
        parts = condition.split("AND")
        return all(self._evaluate_check_condition(data, part.strip(), "") for part in parts)
    
    def _check_or_condition(self, data: Dict[str, Any], condition: str) -> bool:
        """OR条件をチェック"""
        parts = condition.split("OR")
        return any(self._evaluate_check_condition(data, part.strip(), "") for part in parts)
    
    def _check_equality(self, data: Dict[str, Any], condition: str) -> bool:
        """等価条件をチェック"""
        if "!=" in condition:
            parts = condition.split("!=", 1)
            if len(parts) == 2:
                key, value = parts
                return str(data.get(key.strip(), "")) != value.strip()
        elif "==" in condition:
            parts = condition.split("==", 1)
            if len(parts) == 2:
                key, value = parts
                return str(data.get(key.strip(), "")) == value.strip()
        else:
            parts = condition.split("=", 1)
            if len(parts) == 2:
                key, value = parts
                return str(data.get(key.strip(), "")) == value.strip()
        
        # If we can't parse it as equality, treat it as a keyword search
        return self._check_keyword(data, condition)
    
    def _check_comparison(self, data: Dict[str, Any], condition: str) -> bool:
        """比較条件をチェック"""
        # 数値比較のロジック（簡易実装）
        if ">=" in condition:
            parts = condition.split(">=")
        elif "<=" in condition:
            parts = condition.split("<=")
        elif ">" in condition:
            parts = condition.split(">")
        elif "<" in condition:
            parts = condition.split("<")
        else:
            return False
            
        try:
            left_val = float(self._extract_numeric_value(data, parts[0].strip()))
            right_val = float(parts[1].strip())
            
            if ">=" in condition:
                return left_val >= right_val
            elif "<=" in condition:
                return left_val <= right_val
            elif ">" in condition:
                return left_val > right_val
            elif "<" in condition:
                return left_val < right_val
        except (ValueError, TypeError):
            return False
        
        return False
    
    def _check_keyword(self, data: Dict[str, Any], condition: str) -> bool:
        """キーワード検索"""
        return self._search_in_data(data, condition)
    
    def _extract_keywords(self, condition: str) -> List[str]:
        """条件からキーワードを抽出"""
        # 簡易的なキーワード抽出
        keywords = re.findall(r'[^\s\=\!\>\<\(\)]+', condition)
        return [k for k in keywords if len(k) > 1]
    
    def _extract_numeric_value(self, data: Dict[str, Any], key: str) -> float:
        """データから数値を抽出"""
        value = data.get(key, 0)
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            numbers = re.findall(r'\d+', value)
            return float(numbers[0]) if numbers else 0
        return 0
    
    def _search_in_data(self, data: Any, keyword: str) -> bool:
        """データ内でキーワードを再帰的に検索"""
        if isinstance(data, dict):
            for key, value in data.items():
                if keyword in str(key) or self._search_in_data(value, keyword):
                    return True
        elif isinstance(data, list):
            for item in data:
                if self._search_in_data(item, keyword):
                    return True
        elif isinstance(data, str):
            return keyword in data
        
        return False
    
    def evaluate_policy_rules(self, role: str, game_name: str, agent_name: str) -> List[Dict[str, Any]]:
        """
        policyルールを評価し、条件に一致しなかったルールのリストを返す
        
        Args:
            role: エージェントの役職
            game_name: ゲーム名
            agent_name: エージェント名
            
        Returns:
            条件に一致しなかったルールのリスト（best_policy）
        """
        policy_data = self.load_policy_file(role)
        best_policy = []
        
        for rule in policy_data.get("rules", []):
            # IF条件を評価
            if not self._evaluate_rule_condition(rule, game_name, agent_name):
                # 条件に一致しない場合、best_policyに追加
                best_policy.append({
                    "id": rule.get("id"),
                    "name": rule.get("name"),
                    "condition": rule.get("condition", {}).get("if", ""),
                    "then": rule.get("then", ""),
                    "supplement": rule.get("supplement", ""),
                    "priority": rule.get("priority", 0)
                })
        
        return best_policy
    
    def _evaluate_rule_condition(self, rule: Dict[str, Any], game_name: str, agent_name: str) -> bool:
        """
        個別ルールの条件を評価
        
        Args:
            rule: ルール情報
            game_name: ゲーム名
            agent_name: エージェント名
            
        Returns:
            条件が満たされているかどうか
        """
        condition = rule.get("condition", {})
        file_checks = condition.get("file_checks", [])
        
        # 全てのfile_checksが満たされている場合のみTrue
        for check in file_checks:
            if not self.evaluate_file_check(check, game_name, agent_name):
                return False
                
        return True