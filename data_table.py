"""
HW-2 Data Table implementation.

NAME: Nick Wunderle
DATE: Fall 2023
CLASS: CPSC 322

"""

import csv
import tabulate


class DataRow:
    """A basic representation of a relational table row. The row maintains
    its corresponding column information.

    """
    
    def __init__(self, columns=[], values=[]):
        """Create a row from a list of column names and data values.
           
        Args:
            columns: A list of column names for the row
            values: A list of the corresponding column values.

        Notes: 
            The column names cannot contain duplicates.
            There must be one value for each column.

        """
        if len(columns) != len(set(columns)):
            raise ValueError('duplicate column names')
        if len(columns) != len(values):
            raise ValueError('mismatched number of columns and values')
        self.__columns = columns.copy()
        self.__values = values.copy()

        
    def __repr__(self):
        """Returns a string representation of the data row (formatted as a
        table with one row).

        Notes: 
            Uses the tabulate library to pretty-print the row.

        """
        return tabulate.tabulate([self.values()], headers=self.columns())

        
    def __getitem__(self, column):
        """Returns the value of the given column name.
        
        Args:
            column: The name of the column.

        """
        if column not in self.columns():
            raise IndexError('bad column name')
        return self.values()[self.columns().index(column)]


    def __setitem__(self, column, value):
        """Modify the value for a given row column.
        
        Args: 
            column: The column name.
            value: The new value.

        """
        if column not in self.columns():
            raise IndexError('bad column name')
        self.__values[self.columns().index(column)] = value


    def __delitem__(self, column):
        """Removes the given column and corresponding value from the row.

        Args:
            column: The column name.

        """
        if column not in self.columns():
            raise IndexError('no such column')
        del self.__values[self.columns().index(column)]
        del self.__columns[self.columns().index(column)]

    
    def __eq__(self, other):
        """Returns true if this data row and other data row are equal.

        Args:
            other: The other row to compare this row to.

        Notes:
            Checks that the rows have the same columns and values.

        """
        if isinstance(other, DataRow):
            if self.__columns == other.__columns:
                return self.__values == other.__values
        return False

    
    def __add__(self, other):
        """Combines the current row with another row into a new row.
        
        Args:
            other: The other row being combined with this one.

        Notes:
            The current and other row cannot share column names.

        """
        if not isinstance(other, DataRow):
            raise ValueError('expecting DataRow object')
        if len(set(self.columns()).intersection(other.columns())) != 0:
            raise ValueError('overlapping column names')
        return DataRow(self.columns() + other.columns(),
                       self.values() + other.values())


    def columns(self):
        """Returns a list of the columns of the row."""
        return self.__columns.copy()


    def values(self, columns=None):
        """Returns a list of the values for the selected columns in the order
        of the column names given.
           
        Args:
            columns: The column values of the row to return. 

        Notes:
            If no columns given, all column values returned.

        """
        if columns is None:
            return self.__values.copy()
        if not set(columns) <= set(self.columns()):
            raise ValueError('duplicate column names')
        return [self[column] for column in columns]


    def select(self, columns=None):
        """Returns a new data row for the selected columns in the order of the
        column names given.

        Args:
            columns: The column values of the row to include.
        
        Notes:
            If no columns given, all column values included.

        """
        if columns is None:
            return DataRow(self.__columns, self.__values)
        return DataRow(columns, self.values(columns))        

    
    def copy(self):
        """Returns a copy of the data row."""
        return self.select()

    

class DataTable:
    """A relational table consisting of rows and columns of data.

    Note that data values loaded from a CSV file are automatically
    converted to numeric values.

    """
    
    def __init__(self, columns=[]):
        """Create a new data table with the given column names

        Args:
            columns: A list of column names. 

        Notes:
            Requires unique set of column names. 

        """
        if len(columns) != len(set(columns)):
            raise ValueError('duplicate column names')
        self.__columns = columns.copy()
        self.__row_data = []


    def __repr__(self):
        """Return a string representation of the table.
        
        Notes:
            Uses tabulate to pretty print the table.

        """
        data = [self.__columns] + [row.values() for row in self.__row_data]
        return tabulate.tabulate(data)

    
    def __getitem__(self, row_index):
        """Returns the row at row_index of the data table.
        
        Notes:
            Makes data tables iterable over their rows.

        """
        return self.__row_data[row_index]

    
    def __delitem__(self, row_index):
        """Deletes the row at row_index of the data table.

        """
        if row_index >= self.row_count() or row_index < 0:
            raise IndexError('no such index')
        del self.__row_data[row_index]

        
    def load(self, filename, delimiter=','):
        """Add rows from given filename with the given column delimiter.

        Args:
            filename: The name of the file to load data from
            delimeter: The column delimiter to use

        Notes:
            Assumes that the header is not part of the given csv file.
            Converts string values to numeric data as appropriate.
            All file rows must have all columns.
        """
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            num_cols = len(self.columns())
            for row in reader:
                row_cols = len(row)                
                if num_cols != row_cols:
                    raise ValueError(f'expecting {num_cols}, found {row_cols}')
                converted_row = []
                for value in row:
                    converted_row.append(DataTable.convert_numeric(value.strip()))
                self.__row_data.append(DataRow(self.columns(), converted_row))

                    
    def save(self, filename, delimiter=','):
        """Saves the current table to the given file.
        
        Args:
            filename: The name of the file to write to.
            delimiter: The column delimiter to use. 

        Notes:
            File is overwritten if already exists. 
            Table header not included in file output.
        """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter, quotechar='"',
                                quoting=csv.QUOTE_NONNUMERIC)
            for row in self.__row_data:
                writer.writerow(row.values())


    def column_count(self):
        """Returns the number of columns in the data table."""
        return len(self.__columns)


    def row_count(self):
        """Returns the number of rows in the data table."""
        return len(self.__row_data)


    def columns(self):
        """Returns a list of the column names of the data table."""
        return self.__columns.copy()


    def append(self, row_values):
        """Adds a new row to the end of the current table. 

        Args:
            row_data: The row to add as a list of values.
        
        Notes:
            The row must have one value per column. 
        """
        if not isinstance(row_values, list):
            raise TypeError('expected a list to append')
        if len(row_values) != self.column_count():
            raise ValueError('wrong number of values')    
        self.__row_data.append(DataRow(self.columns(),row_values))   

    
    def rows(self, row_indexes):
        """Returns a new data table with the given list of row indexes. 

        Args:
            row_indexes: A list of row indexes to copy into new table.
        
        Notes: 
            New data table has the same column names as current table.

        """
        table = DataTable(self.columns())
        for i in row_indexes:
            if i >= self.row_count() or i < 0:
                raise IndexError('index out of range')
            table.append(self[i].values())
        return table
        
        
    def drop(self, columns):
        """Removes the given columns from the current table.
        
        Args:
            columns: the names of the columns to drop.
        """
        for col in columns:
            if col in self.__columns:
                self.__columns.remove(col)
            for row in self.__row_data:
                if col in row.columns():
                    del row[col]
    
    
    def copy(self):
        """Returns a copy of the current table."""
        table = DataTable(self.columns())
        for row in self:
            table.append(row.values())
        return table
    

    def update(self, row_index, column, new_value):
        """Changes a column value in a specific row of the current table.

        Args:
            row_index: The index of the row to update.
            column: The name of the column whose value is being updated.
            new_value: The row's new value of the column.

        Notes:
            The row index and column name must be valid. 

        """
        if row_index < 0 or row_index > self.row_count():
            raise IndexError('index out of range')
        if column not in self.columns():
            raise IndexError('no such column')
        self.__row_data[row_index][column] = new_value

  
    @staticmethod
    def combine(table1, table2, columns=[], non_matches=False):
        """Returns a new data table holding the result of combining table 1 and 2.

        Args:
            table1: First data table to be combined.
            table2: Second data table to be combined.
            columns: List of column names to combine on.
            nonmatches: Include non matches in answer.

        Notes:
            If columns to combine on are empty, performs all combinations.
            Column names to combine are must be in both tables.
            Duplicate column names removed from table2 portion of result.

        """
        if len(columns) != len(set(columns)):
            raise IndexError('cannot combine on duplicate columns')
        for col in columns:
            if col not in table1.columns() or col not in table2.columns():
                raise IndexError('at least one column is not in both tables')
        new_columns = table1.columns() + [col for col in table2.columns() if col not in columns]
        new_table = DataTable(list(set(new_columns)))
        table1_rows = {}
        table2_rows = {}
        for row in table1.__row_data:
            table1_key = ''
            for col in columns:
                table1_key = table1_key+str(row.__getitem__(col))
            if table1_key in table1_rows:
                table1_rows[table1_key].append(row.values())
            else:
                table1_rows[table1_key] = [row.values()]   
        for row in table2.__row_data:
            table2_key = ''
            for col in columns:
                table2_key = table2_key+str(row.__getitem__(col))
            if table2_key in table2_rows:
                table2_rows[table2_key].append(row.values())
            else:
                table2_rows[table2_key] = [row.values()]
            if table2_key in table1_rows:
                for matching in table1_rows[table2_key]:
                    new_row = matching + row.select([col for col in row.columns() if col not in columns]).values()
                    new_table.append(new_row)
            elif non_matches:
                for non_matching in table2_rows[table2_key]:
                    new_row = ['' for col in new_columns]
                    for i in range(len(new_columns)):
                        for col in table2.columns():
                            if new_columns[i] == col:
                                new_row[i] = row.__getitem__(col)
                    new_table.append(new_row)
        if non_matches:
            for row in table1.__row_data:
                table1_key = ''
                for col in columns:
                    table1_key = table1_key+str(row.__getitem__(col))             
                if table1_key not in table2_rows:
                    for non_matching in table1_rows[table1_key]:
                        new_row = ['' for col in new_columns]
                        for i in range(len(new_columns)):
                            for col in table1.columns():
                                if new_columns[i] == col:
                                    new_row[i] = row.__getitem__(col)
                        new_table.append(new_row)
        return new_table
        

    @staticmethod
    def convert_numeric(value):
        """Returns a version of value as its corresponding numeric (int or
        float) type as appropriate.

        Args:
            value: The string value to convert

        Notes:
            If value is not a string, the value is returned.
            If value cannot be converted to int or float, it is returned.

         """
        converted = None
        try:
            converted = int(value)
        except:
            pass
        if isinstance(converted, int):
            return converted
        else:
            try:
                converted = float(value)
            except:
                converted = value
        return converted
